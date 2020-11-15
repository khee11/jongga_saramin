python preprocess.py --data agnews --tokenizer word --seed 1234 --num_label 150 --num_eval 950 --num_unlab 1500 --min_freq 1
python write_tfrecord.py --data agnews --tokenizer word

