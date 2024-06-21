#!/bin/bash

python visualize.py --test_path datasets/cstnet-tesla-tls1.3/packet/train_dataset.pkl \
                    --labels_num 136 \
                    --load_model_path models/finetuned_cstesla_multi.bin
