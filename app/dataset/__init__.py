import logging
import pickle

import torch
from sklearn.model_selection import train_test_split

from .balanced import BalancedSubset
from .finetune import FineTuneDataset

LOGGER = logging.getLogger("Dataset")


def load_and_split_data(path, test_size=0.2, valid_size=0.1):
    LOGGER.info(
        f"Reading dataset with `valid size = {valid_size}, test size = {test_size}")
    # 加载整个数据集
    data = pickle.load(open(path, 'rb'))
    # 处理数据格式
    if len(data[0]) == 3:
        data = [(torch.tensor(src, dtype=torch.long),
                 torch.tensor(tgt, dtype=torch.long),
                 torch.tensor(seq, dtype=torch.long)) for src, tgt, seq in data]
    else:
        data = [(torch.tensor(src, dtype=torch.long),
                 torch.tensor(tgt, dtype=torch.long),
                 torch.tensor([1] * len(src), dtype=torch.long)) for src, tgt in data]

    # 划分测试集
    train_valid, test_data = train_test_split(
        data, test_size=test_size, random_state=42)
    # 划分验证集
    train_data, valid_data = train_test_split(
        train_valid, test_size=valid_size, random_state=42)

    return train_data, valid_data, test_data
