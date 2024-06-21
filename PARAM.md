# 模型定义

``` python
class Classifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embedding = WordPosSegEmbedding(args, len(args.tokenizer.vocab))
        self.fc_1 = nn.Linear(768, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc_2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc_3 = nn.Linear(256, args.labels_num)

    def forward(self, src, seg):
        o1 = self.embedding(src, seg)   # [batch_size, seq_length, 768]
        o1 = torch.mean(o1, dim=1)      # [batch_size, 768]


        o2 = torch.tanh(self.fc_1(o1))
        o2 = self.dropout1(o2)

        o3 = torch.tanh(self.fc_2(o2))
        o3 = self.dropout2(o3)
        
        logits = self.fc_3(o3)
        return logits
```


# 结果记录

## CSTNET 120class

lr=0.01, seed=42, dropout=0.5 不行

lr=0.002最高到60% 0.001到90%合适

参数添加 --save_model 才保存模型

三层线性层，添加dropout，最好的test precision 91.3% 12轮收敛 /home/natsuu/ET-BERT/results/cstnet/2024-04-23_13-51-07_lr0.002_epoch_20_bs128

## CSTNET-TESLA-二分类

两层线性层第一轮就收敛 /home/natsuu/ET-BERT/results/2024-04-23_14-18-39_lr0.002_epoch_3_bs128

一层也是第一轮就收敛 /home/natsuu/ET-BERT/results/2024-04-23_16-20-56_lr0.002_epoch_3_bs128

etbert 论文中97.3% 复现最好95%

## ISCX-VPN-SERVICE

（数据集太小了）去除encoder，一层线性层，随便跑5轮 98%

默认参数etbert 5轮 98.98%

使用etbert代码里uer的optimizer和scheduler效果没有自己写的好，batch size对测试结果没有明显的正向/负向影响

## USTC-TFC

三层线性层，2e-3 2e-4两三轮就收敛 98.9%
默认参数etbert 5轮 98.89%

## 纯Tesla

三层线性层：样本数太少难收敛
两层线性层：预测准确率 99.65%
一层线性层：99.7%
etbert： 99.46% （与训练策略有关？）
