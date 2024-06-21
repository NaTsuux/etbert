import random

from torch.utils.data import Dataset


class BalancedSubset(Dataset):
    def __init__(self, dataset, num_labels, num_samples=1024):
        self.dataset = dataset
        self.num_labels = num_labels
        self.num_samples = num_samples

        self.indices = [[i for i, (src, tgt, seg) in enumerate(
            self.dataset) if tgt == lb] for lb in range(num_labels)]

        # 随机选取样本
        selected_indices = [random.sample(
            self.indices[i], num_samples) for i in range(num_labels)]

        # 合并并打乱选取的索引
        self.indices = []
        for ind in selected_indices:
            self.indices.extend(ind)

        random.shuffle(self.indices)

    def __len__(self):
        return self.num_labels * self.num_samples

    def __getitem__(self, idx):
        # 直接使用原始数据集的获取方式
        return self.dataset[self.indices[idx]]
