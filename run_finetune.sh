PYTHONPATH=./ python fine-tuning.py \
  --vocab_path models/encryptd_vocab.txt \
  --data_path datasets/contest3.pkl \
  --encoder transformer --mask fully_visible --embedding word_pos_seg \
  --report_steps 100 \
  --epochs_num 5 --batch_size 64 --learning_rate 2e-5 \
  --seq_length 128  --labels_num 100
