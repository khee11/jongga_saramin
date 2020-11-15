python train.py --data agnews --step sup --buffer_size1 1200 --model lstm --batch_size1 128 --epochs 15 --lr 5e-4
python train.py --data agnews --step sup --buffer_size1 1200 --model lstm_base --batch_size1 64 --epochs 15 --lr 5e-479592
python train.py --data agnews --step sup --buffer_size1 1200 --model cnn --batch_size1 64 --epochs 10 --lr 5e-4
python train.py --data agnews --step sup --buffer_size1 1200 --model swem_cat --batch_size1 64 --epochs 50 --lr 1e-3 --epochs 50

python train.py --data agnews --step at --buffer_size1 1200 --model lstm --batch_size1 64 --epochs 20 --eta 2.0 --lr 5e-4
python train.py --data agnews --step at --buffer_size1 1200 --model lstm_base --batch_size1 64 --epochs 15 --eta 2.0 --lr 5e-4 --start_weight 0.40 --end_weight 0.50
python train.py --data agnews --step at --buffer_size1 1200 --model cnn --batch_size1 64 --epochs 20 --eta 2.0 --lr 5e-4
python train.py --data agnews --step at --buffer_size1 1200 --model swem_cat --batch_size1 64 --epochs 50 --eta 2.0 --lr 5e-4

python train.py --data agnews --step pi --model lstm --epochs 30 --buffer_size1 1200 --buffer_size2 12000 --lr 5e-4 --start_weight 0.35 --end_weight 0.50
python train.py --data agnews --step pi --model lstm_base --epochs 10 --buffer_size1 1200 --buffer_size2 12000 --lr 5e-4 --start_weight 0.35 --end_weight 0.40 --eta 2.0
python train.py --data agnews --step pi --model cnn --epochs 30  --buffer_size1 1200 --buffer_size2 12000 --lr 5e-4 --start_weight 0.35 --end_weight 0.50
python train.py --data agnews --step pi --model swem_cat --epochs 40  --buffer_size1 1200 --buffer_size2 12000 --lr 5e-4 --start_weight 0.35 --end_weight 0.50

python train.py --data agnews --step vat --model lstm --epochs 30  --batch_size1 16 --batch_size2 128 --buffer_size1 1200 --buffer_size2 12000 --lr 5e-4 --start_weight 0.35 --end_weight 0.50 --eta 2.0
python train.py --data agnews --step vat --model lstm_base --epochs 20  --buffer_size1 1200 --buffer_size2 12000 --lr 5e-4 --start_weight 0.35 --end_weight 0.40 --eta 2.0
python train.py --data agnews --step vat --model cnn --epochs 30 --buffer_size1 1200 --buffer_size2 12000   --lr 5e-4 --start_weight 0.35 --end_weight 0.50
python train.py --data agnews --step vat --model swem_cat --epochs 40 --buffer_size1 1200 --buffer_size2 12000  --lr 5e-4 --start_weight 0.35 --end_weight 0.50

python train.py --data agnews --step both --model lstm --epochs 20 --buffer_size1 1200 --buffer_size2 12000  --lr 5e-4 --start_weight 0.35 --end_weight 0.50
python train.py --data agnews  --step both --model lstm_base --epochs 20 --buffer_size1 1200 --buffer_size2 12000  --lr 5e-4 --start_weight 0.35 --end_weight 0.50 --eta 2.0
python train.py --data agnews --step both --model cnn --epochs 20 --buffer_size1 1200 --buffer_size2 12000  --lr 5e-4 --start_weight 0.35 --end_weight 0.50
python train.py --data agnews --step both --model swem_cat --epochs 35 --buffer_size1 1200 --buffer_size2 12000  --lr 5e-4 --start_weight 0.35 --end_weight 0.50



