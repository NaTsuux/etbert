"""
This script provides an exmaple to wrap UER-py for classification.
"""

import argparse
import json
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from app import init_logger
from app.dataset import FineTuneDataset, load_and_split_data
from app.model.classifier import Classifier
from uer.encoders import *
from uer.layers import *
from uer.model_saver import save_model
from uer.opts import finetune_opts
from uer.utils import *
from uer.utils.config import load_hyperparam
from uer.utils.constants import *
from uer.utils.optimizers import *
from uer.utils.seed import set_seed

TQDM_FORMAT = {
    "ncols": 0,
    "bar_format": "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
}


def load_or_initialize_parameters(args, model):
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model.load_state_dict(
            torch.load(
                args.pretrained_model_path,
                map_location={
                    "cuda:1": "cuda:0",
                    "cuda:2": "cuda:0",
                    "cuda:3": "cuda:0",
                },
            ),
            strict=False,
        )
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)


def build_optimizer(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay_rate": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay_rate": 0.0,
        },
    ]
    if args.optimizer in ["adamw"]:
        optimizer = str2optimizer[args.optimizer](
            optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False
        )
    else:
        optimizer = str2optimizer[args.optimizer](
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            scale_parameter=False,
            relative_step=False,
        )
    if args.scheduler in ["constant"]:
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](
            optimizer, args.train_steps * args.warmup
        )
    else:
        scheduler = str2scheduler[args.scheduler](
            optimizer, args.train_steps * args.warmup, args.train_steps
        )
    return optimizer, scheduler


def train_model(
    args,
    model,
    optimizer,
    scheduler,
    src_batch,
    tgt_batch,
    seg_batch,
    soft_tgt_batch=None,
):
    model.zero_grad()

    loss, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch)
    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)

    if args.fp16:
        with args.amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    optimizer.step()
    scheduler.step()

    return loss


def evaluate(args, dataloader: DataLoader, print_confusion_matrix=False):
    confusion = torch.zeros(
        args.labels_num, args.labels_num, dtype=torch.long, device=args.device
    )
    total_samples = 0  # 总样本数
    correct = 0  # 正确预测数

    args.model.eval()

    for src_batch, tgt_batch, seg_batch in tqdm.tqdm(
        dataloader, desc="Evaluating...", **TQDM_FORMAT
    ):
        src_batch, tgt_batch, seg_batch = (
            src_batch.to(args.device),
            tgt_batch.to(args.device),
            seg_batch.to(args.device),
        )
        with torch.no_grad():
            _, logits = args.model(src_batch, tgt_batch, seg_batch)
        pred = torch.argmax(F.softmax(logits, dim=1), dim=1)
        gold = tgt_batch.flatten()  # [32, 1] to [32, ]
        for j in range(pred.size(0)):
            confusion[pred[j], gold[j]] += 1
        correct += torch.sum(pred == gold).item()

        total_samples += tgt_batch.size(0)  # 累计样本数

    try:
        if print_confusion_matrix:
            logging.info("Confusion matrix:")
            logging.info(confusion)
            cf_array = confusion.cpu().numpy()
            with open("./results/confusion_matrix.txt", "w") as f:
                for cf_a in cf_array:
                    f.write(str(cf_a) + "\n")
            logging.info("Report precision, recall, and f1:")
            eps = 1e-9
            for i in range(args.labels_num):
                p = confusion[i, i].item() / \
                    (confusion[i, :].sum().item() + eps)
                r = confusion[i, i].item() / \
                    (confusion[:, i].sum().item() + eps)
                f1 = 2 * p * r / (p + r + eps) if (p + r) > 0 else 0
                logging.info(f"Label {i}: P={p:.3f}, R={r:.3f}, F1={f1:.3f}")

        logging.info(
            f"Acc. (Correct/Total): {correct / total_samples:.10f} ({correct}/{total_samples}) "
        )
    except Exception as e:
        return 0, []
    return correct / total_samples, confusion


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    finetune_opts(parser)

    parser.add_argument(
        "--pooling",
        choices=["mean", "max", "first", "last"],
        default="first",
        help="Pooling type.",
    )

    parser.add_argument(
        "--tokenizer",
        choices=["bert", "char", "space"],
        default="bert",
        help="Specify the tokenizer."
        "Original Google BERT uses bert tokenizer on Chinese corpus."
        "Char tokenizer segments sentences into characters."
        "Space tokenizer segments sentences into words according to space.",
    )

    parser.add_argument(
        "--soft_targets", action="store_true", help="Train model with logits."
    )
    parser.add_argument(
        "--soft_alpha", type=float, default=0.5, help="Weight of the soft targets loss."
    )

    parser.add_argument(
        "--labels_num", type=int, required=True, help="label count of train set"
    )
    parser.add_argument(
        "--save_model", type=bool, default=False
    )

    args = parser.parse_args()
    args.result_dir = "./results"
    init_logger(args)

    logging.info(json.dumps(args.__dict__, indent=4))

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Build classification model.
    model = Classifier(args)

    # Load or initialize parameters.
    load_or_initialize_parameters(args, model)

    args.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)

    batch_size = args.batch_size

    train_data, valid_data, test_data = load_and_split_data(args.data_path)

    # Preparing dataset
    train_loader = DataLoader(
        FineTuneDataset(train_data), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    valid_loader = DataLoader(
        FineTuneDataset(valid_data), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    test_loader = DataLoader(
        FineTuneDataset(test_data), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    instances_num = len(train_data)
    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    logging.info("Batch size: %d", batch_size)
    logging.info("The number of training instances: %d", instances_num)

    # Preparing optimizer
    optimizer, scheduler = build_optimizer(args, model)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level
        )
        args.amp = amp

    if torch.cuda.device_count() > 1:
        logging.info(
            "{} GPUs are available. Let's use them.".format(
                torch.cuda.device_count())
        )
        model = torch.nn.DataParallel(model)
    args.model = model

    total_loss, result, best_result = 0.0, 0.0, 0.0

    writer = SummaryWriter(args.result_dir)

    # Training phase
    logging.info("Start training.")
    for epoch in range(1, args.epochs_num + 1):
        logging.info(f"Epoch #{epoch}:")
        model.train()
        for i, (src_batch, tgt_batch, seg_batch) in tqdm.tqdm(
                enumerate(train_loader), desc="Training", total=len(train_loader)):
            src_batch, tgt_batch, seg_batch = (
                src_batch.to(args.device),
                tgt_batch.to(args.device),
                seg_batch.to(args.device),
            )

            loss = train_model(
                args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch
            )
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                logging.info(
                    "Epoch id: {}, Training steps: {}, Avg loss: {:.10f}".format(
                        epoch, i + 1, total_loss / args.report_steps
                    )
                )
                writer.add_scalar("Loss/train", total_loss /
                                  args.report_steps, i)
                total_loss = 0.0
        logging.info(f"Epoch #{epoch} training done. Evaluating: ")

        result = evaluate(args, valid_loader)
        if args.save_model and result[0] > best_result:
            logging.info(
                f"Best result, save model to {args.output_model_path}")
            best_result = result[0]
            save_model(model, args.output_model_path)

    # Evaluation phase.
    logging.info("Test set evaluation.")
    evaluate(args, test_loader, True)
    writer.close()


if __name__ == "__main__":

    main()
