import argparse
import binascii
import os
import pickle
import random
import subprocess
from pathlib import Path

import tqdm
from scapy.all import PcapReader
from scapy.layers.inet import TCP, UDP

from uer.utils.constants import *
from uer.utils.tokenizers import BertTokenizer


def remove_duplicate(base: Path, to_base: Path):
    """通过editcap去除tcp retransmission"""
    os.makedirs(to_base, exist_ok=True)
    for f in base.iterdir():
        to_file = to_base / f.name
        o = subprocess.run(
            ["editcap", "-D", "100",
                str(f.absolute()), str(to_file.absolute())],
            capture_output=True,
        )
        # print(o.stdout)


def generate_raw_dataset(args, pcap_dir: Path):
    """读取初步清理后的pcap目录，并生成数据集
    - `label` 取决于 `pcap_dir` 目录内的目录划分
    - 对于每个 label 目录，读取其中所有 pcap 文件，并对其所有 packet 做数据处理
    - 生成的 `dataset.pkl` 保存在 `pcap_dir` 目录下，可直接用于 `FineTuneDataset`
    """
    labels_dir = [x for x in pcap_dir.iterdir() if "." not in x.name]
    labels = [x.name for x in labels_dir]
    print(f"labels: {labels}, count {len(labels)}")

    out_dir = pcap_dir / "../raw"
    out_dir.mkdir(exist_ok=True)

    # define bigram generation function
    def bigram_generation(packet_datagram, packet_len=64):
        def cut(obj, sec):
            sec = sec % 4 + sec
            return [obj[i: i + sec] for i in range(0, len(obj), sec)]

        result = ""
        generated_datagram = cut(packet_datagram, 1)
        token_count = 0
        for sub_string_index in range(len(generated_datagram)):
            if sub_string_index != (len(generated_datagram) - 1):
                token_count += 1
                if token_count > packet_len:
                    break
                else:
                    merge_word_bigram = (
                        generated_datagram[sub_string_index]
                        + generated_datagram[sub_string_index + 1]
                    )
            else:
                break
            result += merge_word_bigram
            result += " "

        return result

    # Begin
    for idx, ld in enumerate(labels_dir):
        fs = [x for x in ld.iterdir()]
        data = []

        with tqdm.tqdm(total=len(fs), desc=f"Parsing {str(ld)}") as pbar:
            for f in fs:
                pbar.update(1)
                pbar.set_description(
                    f"Parsing {str(ld)}, current len(data): {len(data)}"
                )
                if 'pcap' not in f.name:
                    continue

                # packets = [
                #     p for p in packets if TCP in p and len(p[TCP].payload) >= 128
                # ]

                x = []

                for packet in PcapReader(f.as_posix()):
                    if not packet.haslayer(TCP):
                        continue
                    packet_data = packet[TCP].payload

                    payload_string = binascii.hexlify(
                        bytes(packet_data)).decode()
                    xx = bigram_generation(payload_string, packet_len=128)
                    x.append(xx)

                    if len(x) >= 10:
                        break

                data.append(''.join(x))
        pickle.dump({"label": idx, "data": data},
                    (out_dir / f"{ld.name}.pkl").open("wb"))


def generate_dataset(args, raw_dataset_dir: Path):
    data = []
    for d in raw_dataset_dir.iterdir():
        ds = pickle.load(d.open("rb"))
        # ds = pickle.load((d / "dataset_raw.pkl").open("rb"))

        print(f"Length of dataset {d.name}: {len(ds['data'])}")
        lb_data = random.sample(
            ds["data"], args.samples) if args.samples is not None else ds["data"]

        for xx in tqdm.tqdm(
            lb_data, total=len(lb_data), desc=f"Tokenizing dataset {d.name}"
        ):
            src = tokenizer.convert_tokens_to_ids(
                [CLS_TOKEN] + tokenizer.tokenize(xx))

            if len(src) > args.seq_length:
                src = src[: args.seq_length]
            else:
                padding = [0] * (args.seq_length - len(src))
                src.extend(padding)

            data.append((src, ds["label"]))

    random.shuffle(data)
    pickle.dump(data, (raw_dataset_dir / "dataset.pkl").open("wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--file_path", default="/home/natsuu/ET-BERT/datasets/pcap", type=str
    )
    parser.add_argument(
        "--vocab_path", default="models/encryptd_vocab.txt", type=str)
    parser.add_argument(
        "--spm_model_path",
        default=None,
        type=str,
    )
    parser.add_argument("--seq_length", type=int,
                        default=512, help="Sequence length.")
    parser.add_argument("--samples", type=int, default=None)

    args = parser.parse_args()
    tokenizer = BertTokenizer(args)

    generate_raw_dataset(args, Path(args.file_path))
    generate_dataset(args, Path(args.file_path) / "../raw")
