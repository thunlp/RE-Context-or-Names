This directory contains code and data for pre-training step.

### 1. Dataset 

You can download our dataset from [google drive]() or [Tsinghua cloud](https://cloud.tsinghua.edu.cn/f/f55fd09903c94baa9436/?dl=1). And then place the dataset in `./data` directory. (You may need `mkdir data`)

### 2. Pretrained Model

You can download our pretrained model MTB from [google drive]() or [Tsinghua cloud](https://cloud.tsinghua.edu.cn/f/5ce773cc67294ce488e5/?dl=1), CP from [google drive]() or [Tsinghua cloud](https://cloud.tsinghua.edu.cn/f/4097d1055962483cb6d9/?dl=1). And then place them in `./ckpt` directory.(You may need `mkdir ckpt`).

### 3. Pretrain
Pretrain MTB:
```shell
python -m torch.distributed.launch --nproc_per_node 4  main.py \
	--cuda 4,5,6,7 \
	--model MTB \
	--lr 3e-5 --batch_size_per_gpu 32 --max_epoch 20 \
	--gradient_accumulation_steps 2 \
	--max_length 64 \
	--save_step 5000 \
	--alpha 0.3 \
	--train_sample \
	--save_dir ckpt_mtb \
```
Pretrain CP:

```shell
python -m torch.distributed.launch --nproc_per_node 4  main.py \
	--cuda 4,5,6,7 \
	--model MTB \
	--lr 3e-5 --batch_size_per_gpu 32 --max_epoch 20 \
	--gradient_accumulation_steps 16 \
	--max_length 64 \
	--save_step 500 \
	--alpha 0.3 \
	--temperature 0.05 \
	--train_sample \
	--save_dir ckpt_cp \
```
