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

**IMPORTANT**

We don't use our own code to eval the models on SemEval, we use the **official** evaluation script. See https://github.com/sahitya0000/Relation-Classification/tree/master/corpus/SemEval2010_task8_scorer-v1.2. So if the results on SemEval are abnormal, please use the official script. The other datasets can be evaluated normally using this code.

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

**IMPORTANT**

We don't provide test set. If you want to test your model, please upload your results to https://thunlp.github.io/fewrel.html. See https://github.com/thunlp/FewRel. 