This directory contains code and data for downstream tasks(Supervised RE and Fewshot RE).

### 1. Supervised RE

#### 1.1 Dataset

We provide SemEval, Wiki80, ChemProt in `data/`, but you need download TACRED from [LDC](https://catalog.ldc.upenn.edu/LDC2018T24) manually. 

Please ensure every dataset has `train.txt`, `dev.txt`,`test.txt`and `rel2id.json`(**NA must be 0 if this benchmark has NA relation**). And `train.txt`(the same as `dev.txt`, `text.txt`) should have multiple lines, each line has the following json-format:

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

#### 1.2 Train
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

"None" means Bert. You can use any checkpoint in `../pretrain/ckpt` directory for finetuning.

#### 2. FewShot RE

We have downloaded [FewRel](https://github.com/thunlp/FewRel) into `fewshotRE` and modified some lines.

Run the following scirpt:

```shell
bash run.sh
```

If you want to use different models, you can change `ckpt` in `run.sh`

```shell
array=(42)
path="paht/to/ckpt"
for seed in ${array[@]}
do
	bash train.sh 7 $seed $path
done
```

"None" means Bert. You can use any checkpoint in `../pretrain/ckpt` directory for finetuning.

