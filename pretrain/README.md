This directory contains code and data for pre-training step.

### 1. Dataset 

You can download our dataset from [google drive](https://drive.google.com/file/d/1V9C8678G-zudBa2rzFtH1ZeNieh3UFG_/view?usp=sharing) or [Tsinghua cloud](https://cloud.tsinghua.edu.cn/f/0f85a20ff8794642a2a3/?dl=1). And then place the dataset in `./data` directory. (You may need `mkdir data`) And then run `code/prepare_data.py` to prepare pre-training data.

### 2. Pretrained Model

You can download our pretrained model MTB from [google drive](https://drive.google.com/file/d/1viGnWGg3B-LasR9UWQFg3lhl-hOEl5ed/view?usp=sharing) or [Tsinghua cloud](https://cloud.tsinghua.edu.cn/f/5ce773cc67294ce488e5/?dl=1), CP from [google drive](https://drive.google.com/file/d/1WU39lYAkZ9JYXlCZFGyAxBlQ--5IU4m6/view?usp=sharing) or [Tsinghua cloud](https://cloud.tsinghua.edu.cn/f/4097d1055962483cb6d9/?dl=1). And then place them in `./ckpt` directory.(You may need `mkdir ckpt`).

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
