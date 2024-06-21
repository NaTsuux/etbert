import argparse
import logging

import matplotlib.pyplot as plt
import torch
import tqdm
import numpy as np

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from app.dataset import BalancedSubset, FineTuneDataset
from app.model import Classifier
from uer.model_loader import load_model
from uer.opts import infer_opts
from uer.utils import *
from uer.utils.config import load_hyperparam

LOGGER = logging.getLogger("Visualize")
TQDM_FORMAT = {
    "ncols": 0,
    "bar_format": "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
}


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    infer_opts(parser)

    parser.add_argument(
        "--pooling",
        choices=["mean", "max", "first", "last"],
        default="first",
        help="Pooling type.",
    )

    parser.add_argument(
        "--labels_num", type=int, required=True, help="Number of prediction labels."
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

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Build classification model and load parameters.
    args.soft_targets, args.soft_alpha = False, False
    model = Classifier(args)
    model = load_model(model, args.load_model_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    batch_size = args.batch_size

    # load dataset
    testset = FineTuneDataset(args.test_path)
    balanced = BalancedSubset(testset, args.labels_num, 16)
    test_loader = DataLoader(
        balanced, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    model.eval()
    encoder_outputs = []
    all_targets = []

    for i, (src_batch, tgt_batch, seg_batch) in tqdm.tqdm(
        enumerate(test_loader),
        desc="Infering...",
        total=len(test_loader),
        **TQDM_FORMAT
    ):
        src_batch = src_batch.to(device)
        tgt_batch = tgt_batch.to(device)
        seg_batch = seg_batch.to(device)
        with torch.no_grad():
            encoder_output, _ = model(src_batch, None, seg_batch)
        encoder_outputs.append(encoder_output)
        all_targets.append(tgt_batch)

    LOGGER.info("Infer done. Generating pic")
    encoder_outputs = torch.cat(encoder_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # 对特征进行求平均
    encoder_outputs = encoder_outputs.cpu().numpy().mean(axis=1)
    all_targets = all_targets.cpu().numpy().flatten()

    # t-SNE 降维
    tsne = TSNE(n_components=2, random_state=42)
    reduced_data = tsne.fit_transform(encoder_outputs)

    # K-means 聚类
    k = 10  # 选择聚类的数量
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(reduced_data)

    # 绘制聚类结果
    plt.figure(figsize=(20, 15))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='viridis', alpha=0.6, s=10)
    plt.colorbar(scatter, ticks=range(k))
    plt.title("t-SNE projection with K-means Clustering")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    # 聚类包含的原始类别标注
    for i in range(k):
        print(f"Cluster {i}: Labels", np.unique(all_targets[clusters == i]))

    plt.savefig("out.png")
    plt.show()


if __name__ == "__main__":
    main()
