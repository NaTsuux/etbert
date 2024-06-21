import argparse
import json
import logging
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import uer
import uer.utils
from app import init_logger
from app.dataset import FineTuneDataset, load_and_split_data
from uer.layers.embeddings import WordPosSegEmbedding
from uer.opts import finetune_opts
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.utils.tokenizers import BertTokenizer


class Classifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embedding = WordPosSegEmbedding(args, len(args.tokenizer.vocab))

        # self.fc_1 = nn.Linear(768, args.labels_num)

        # self.fc_1 = nn.Linear(768, 512)
        # self.dropout1 = nn.Dropout(0.5)
        # self.fc_2 = nn.Linear(512, args.labels_num)

        self.fc_1 = nn.Linear(768, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc_2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc_3 = nn.Linear(256, args.labels_num)

    def forward(self, src, seg):
        o1 = self.embedding(src, seg)   # [batch_size, seq_length, 768]
        o1 = torch.mean(o1, dim=1)      # [batch_size, 768]

        # logits = self.fc_1(o1)

        o2 = torch.tanh(self.fc_1(o1))
        o2 = self.dropout1(o2)

        logits = self.fc_2(o2)

        o3 = torch.tanh(self.fc_2(o2))
        o3 = self.dropout2(o3)
        logits = self.fc_3(o3)
        return logits


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
    optimizer = uer.utils.AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    scheduler = uer.utils.get_linear_schedule_with_warmup(
        optimizer, args.train_steps * args.warmup, args.train_steps)

    return optimizer, scheduler


def validate(model, device, loader, labels_num, print_matrix=False):
    confusion = torch.zeros(
        labels_num, labels_num, dtype=torch.long, device=device
    )
    model.eval()
    correct, total_samples = 0, 0

    for src_batch, tgt_batch, seg_batch in tqdm.tqdm(loader, desc="Evaluating", total=len(loader)):
        src_batch, tgt_batch, seg_batch = src_batch.to(
            device), tgt_batch.to(device), seg_batch.to(device)
        with torch.no_grad():
            logits = model(src_batch, seg_batch)
            pred = torch.argmax(logits, dim=1)
            gold = tgt_batch.flatten()
        for j in range(pred.size(0)):
            confusion[pred[j], gold[j]] += 1

        correct += torch.sum(pred == tgt_batch).item()
        total_samples += tgt_batch.size(0)
    accuracy = correct / total_samples
    logging.info(
        f"Acc. (Correct/Total): {correct / total_samples:.10f} ({correct}/{total_samples}) "
    )

    if print_matrix:
        logging.info(confusion)
        cf_array = confusion.cpu().numpy()
        with open("./results/confusion_matrix.txt", "w") as f:
            for cf_a in cf_array:
                f.write(str(cf_a) + "\n")
        logging.info("Report precision, recall, and f1:")
        eps = 1e-9
        for i in range(labels_num):
            p = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)
            r = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)
            f1 = 2 * p * r / (p + r + eps) if (p + r) > 0 else 0
            logging.info(f"Label {i}: P={p:.3f}, R={r:.3f}, F1={f1:.3f}")
    return accuracy


def main(args, writer: SummaryWriter):
    set_seed(args.seed)
    load_hyperparam(args)

    logging.info(
        f"Training started with the following parameters: {json.dumps(args.__dict__, indent=4)}")

    args.tokenizer = BertTokenizer(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Classifier(args)
    model.load_state_dict(
        torch.load(args.pretrained_model_path, map_location={
                   "cuda:1": "cuda:0", "cuda:2": "cuda:0", "cuda:3": "cuda:0"}),
        strict=False,
    )
    model.to(device)

    train_data, valid_data, test_data = load_and_split_data(
        args.data_path, 0.1, 0.1)
    train_loader = DataLoader(FineTuneDataset(
        train_data), batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(FineTuneDataset(
        valid_data), batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(FineTuneDataset(
        test_data), batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    logging.info(
        f"Training data size: {len(train_data)}, Validation data size: {len(valid_data)}")

    # OPTIMIZER
    args.train_steps = int(
        len(train_data) * args.epochs_num / args.batch_size) + 1
    # optimizer, scheduler = build_optimizer(args, model)
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=args.step_gamma)
    best_result = 0.0

    for epoch in range(1, args.epochs_num + 1):
        logging.info(f"Epoch {epoch} started.")
        model.train()
        total_loss = 0.0
        for i, (src_batch, tgt_batch, seg_batch) in enumerate(tqdm.tqdm(train_loader, desc="Training", total=len(train_loader))):
            src_batch, tgt_batch, seg_batch = src_batch.to(
                device), tgt_batch.to(device), seg_batch.to(device)
            optimizer.zero_grad()
            logits = model(src_batch, seg_batch)
            loss = nn.CrossEntropyLoss()(logits, tgt_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                avg_loss = total_loss / args.report_steps

                writer.add_scalar('Training Loss', avg_loss,
                                  epoch * len(train_loader) + i)
                logging.info(
                    f"Epoch {epoch}, Step {i+1}, Avg Loss: {avg_loss:.10f}")
                total_loss = 0.0

        # change lr after single epoch
        scheduler.step()

        val_accuracy = validate(model, device, valid_loader, args.labels_num)
        writer.add_scalar('Validation Accuracy', val_accuracy, epoch)
        logging.info(
            f"Epoch {epoch}, Validation Accuracy: {val_accuracy:.10f}")

        if val_accuracy > best_result:
            best_result = val_accuracy
            if args.save_model:
                torch.save(model.state_dict(),
                           f"{args.result_dir}/model_best.pth")
    logging.info("Test set evaluation.")
    validate(model, device, test_loader, args.labels_num, True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    finetune_opts(parser)
    parser.add_argument(
        "--labels_num", type=int, required=True, help="label count of train set"
    )
    parser.add_argument(
        "--save_model", type=bool, default=False
    )
    parser.add_argument(
        "--result_dir", type=str, required=True,
    )

    args = parser.parse_args()

    start_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    args.log_file = f"{start_time}_lr{args.learning_rate}_epoch_{args.epochs_num}_bs{args.batch_size}.log"
    # args.result_dir = f"results/{start_time}_lr{args.learning_rate}_epoch_{args.epochs_num}_bs{args.batch_size}"
    os.makedirs(args.result_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=args.result_dir)
    init_logger(args)

    logging.info(" ".join(sys.argv))

    main(args, writer)

    writer.close()
