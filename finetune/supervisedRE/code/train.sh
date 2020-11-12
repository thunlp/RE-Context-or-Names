python main.py --cuda $1 \
	--seed $2 \
	--lr 3e-5 --batch_size_per_gpu 64 --max_epoch $5 \
	--max_length 100 \
	--mode CM \
	--dataset tacred \
	--entity_marker --ckpt_to_load $3 \
	--train_prop $4 \
