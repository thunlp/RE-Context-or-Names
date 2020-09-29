# RE Analysis

Dataset and code for [Learning from Context or Names? An Empirical Study on Neural Relation Extraction](https://github.com/thunlp/RE-Context-or-Names). 

### Quick Start

You can quickly run our code by following steps:

- Install dependencies as described in following section. 
- Download dataset to `./data` directory and download our pretrained model to `./ckpt` directory if you need. 

- Run `bash init.sh` to pre-process data. 
- Pretrain your own model or finetune on different RE datasets.  

### Dependencies

Run the following script to install dependencies.

```shell
pip install -r requirement.txt
```

**However, you need install following dependencies manually.**

#### transformers

We use huggingface transformers to implement Bert, and the version is 2.5.0. You need clone or download [transformers repo](https://github.com/huggingface/transformers). And in  `src/transformers/modeling_bert.py`  class `BertForMaskedLM` function `forward()`, you should add 

```
outputs = (sequence_output,) + outputs
```

before `return outputs`.  And then, use 

```
pip install .
```

to install transformers manually.

 ### Dataset 

You can download our dataset from [google drive]() or [Tsinghua cloud](https://cloud.tsinghua.edu.cn/f/f55fd09903c94baa9436/?dl=1). And then place the dataset in `./data` directory.(You may need `mkdir data`)

If you want use your own dataset to pretrain model, please ensure that the format is the same with our dataset released.

 ### Pretrained Model

You can download our pretrained model MTB from [google drive]() or [Tsinghua cloud](https://cloud.tsinghua.edu.cn/f/5ce773cc67294ce488e5/?dl=1), CP from [google drive]() or [Tsinghua cloud](https://cloud.tsinghua.edu.cn/f/4097d1055962483cb6d9/?dl=1). And then place them in `./ckpt` directory.(You may need `mkdir ckpt`)

### Pretrain

You can use this repo to pretrain a new model. To pretrain MTB:

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

To pretrain CP:

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



### Finetune

##### Supervised RE

Download tacred, wiki80, semeval from [OpenNRE](https://github.com/thunlp/OpenNRE), chemprot from [scibert](https://github.com/allenai/scibert). Please ensure every benchmark has `train.txt`, `dev.txt`,`test.txt`and `rel2id.json`(**NA must be 0 if this benchmark has NA relation**). And `train.txt`(the same as `dev.txt`, `text.txt`) should have multiple lines, each line has the following json-format:

```python
{
    "tokens":["Microsoft", "was", "founded", "by", "Bill", "Gates", "."], 
    "h":{
        "name": "Microsotf", "pos":[0,1]  # Left closed and right open interval
    }
    "t":{
        "name": "Bill Gates", "pos":[4,6] # Left closed and right open interval
    }
    "relation": "founded_by"
}
```

Run the following scirpt:

```shell
bash run.sh
```

If you want to use different model, you can change `ckpt` in `run.sh`

```shell
array=(42 43 44 45 46)
ckpt="None"
for seed in ${array[@]}
do
	bash train.sh 1 $seed $ckpt 1 6
done
```

"None" means Bert. You can use any checkpoint in `ckpt` directory for finetuning.

##### FewShot RE

You need clone or download [FewRel](https://github.com/thunlp/FewRel). And use Bert as encoder and load pretrained model to finetune on the FewShot dataset. 

```python
ckpt = torch.load(path/to/your/ckpt)
bert.load_state_dict(ckpt["bert-base"])
```

