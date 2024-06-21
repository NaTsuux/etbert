import sys
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

from uer.utils.tokenizers import BertTokenizer
from uer.utils.constants import *

def preprocess_tsv(path, tokenizer, seq_length, soft_targets=False, recursive=False):
    path = Path(path)
    if recursive and path.is_dir():
        files = [f for f in path.iterdir() if f.is_file() and f.suffix == ".tsv" and f.stem != "nolabel_test_dataset"]
    else:
        files = [path]

    for f in files:
        fp = f.open('r', encoding='UTF-8')
        title = fp.readline()
        columns = {k: v for v, k in enumerate(title.strip().split("\t"))}

        data = []

        for line in tqdm(fp, desc=f"Parsing {f.name} with columns {[','.join(columns.keys())]}"):
            line = line.strip().split("\t")
            label = int(line[columns["label"]]) if "label" in columns else -1
            text_a = line[columns["text_a"]]

            # TODO pretrain dataset
            src = tokenizer.convert_tokens_to_ids(
                [CLS_TOKEN] + tokenizer.tokenize(text_a)
            )
            seg = [1] * len(src)

            if len(src) > seq_length:
                src = src[: seq_length]
                seg = seg[: seq_length]
            else:
                padding = [0] * (seq_length - len(src))
                src.extend(padding)
                seg.extend(padding)
            
            soft_tgt = [float(value) for value in line[columns["logits"]].split(" ")] if soft_targets and "logits" in columns else None

            data.append((src, label, seg, soft_tgt))
        pickle.dump(data, open(f"{f.parent}/{f.stem}.pkl", "wb"))
        tqdm.write(f"File {f.name} preprocess done.")

def preprocess_npy(path, tokenizer, seq_length):
    path = Path(path)
    data = []

    for pharse in ["train", "test", "valid"]:
        x_np = np.load(path / f"x_datagram_{pharse}.npy")
        y_np = np.load(path / f"y_{pharse}.npy")

        assert len(x_np) == len(y_np)
        for xx, yy in tqdm(zip(x_np, y_np), f"Parsing {pharse} dataset"):
            src = tokenizer.convert_tokens_to_ids(
                [CLS_TOKEN] + tokenizer.tokenize(xx)
            )
            seg = [1] * len(src)

            if len(src) > seq_length:
                src = src[: seq_length]
                seg = seg[: seq_length]
            else:
                padding = [0] * (seq_length - len(src))
                src.extend(padding)
                seg.extend(padding)
            data.append((src, yy, seg))
    pickle.dump(data, (path / f'dataset.pkl').open('wb'))
    print("OK")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--file_type", default="npy", type=str
    )
    parser.add_argument(
        "--file_path", default="/home/natsuu/ET-BERT/datasets/raw/tesla_capture/packet", type=str
    )
    parser.add_argument(
        "--vocab_path", default="models/encryptd_vocab.txt", type=str
    )
    parser.add_argument(
        "--spm_model_path",
        default=None,
        type=str,
    )
    parser.add_argument("--seq_length", type=int, default=128, help="Sequence length.")

    args = parser.parse_args()
    tokenizer = BertTokenizer(args)

    if args.file_type == "tsv":
        preprocess_tsv(args.file_path, tokenizer, args.seq_length, recursive=True)
    elif args.file_type == "npy":
        preprocess_npy(args.file_path, tokenizer, args.seq_length)
