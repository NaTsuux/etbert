PYTHONPATH=. python3 pre-training/pretrain.py --dataset_path dataset.pt --vocab_path models/encryptd_vocab.txt \
                       --pretrained_model_path models/pre-trained_model.bin \
                       --output_model_path models/add_pretrained_model.bin \
                       --gpu_ranks 0 \
                       --total_steps 500000 --save_checkpoint_steps 100000 --batch_size 32 \
                       --embedding word_pos_seg --encoder transformer --mask fully_visible --target bert
