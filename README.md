# RE Analysis

Dataset and code for [Learning from Context or Names? An Empirical Study on Neural Relation Extraction](https://arxiv.org/abs/2010.01923). 

If you use this code, please cite us
```
@article{peng2020learning,
  title={Learning from Context or Names? An Empirical Study on Neural Relation Extraction},
  author={Peng, Hao and Gao, Tianyu and Han, Xu and Lin, Yankai and Li, Peng and Liu, Zhiyuan and Sun, Maosong and Zhou, Jie},
  journal={arXiv preprint arXiv:2010.01923},
  year={2020}
}
```

### Quick Start

You can quickly run our code by following steps:

- Install dependencies as described in following section. 
- cd to `pretrain` or `finetune` directory then download and pre-processing data for pre-traing or finetuning.    


### 1. Dependencies

Run the following script to install dependencies.

```shell
pip install -r requirement.txt
```

**However, you need install transformers manually.**

We use huggingface transformers to implement Bert.  And for convenience, we have downloaded  [transformers](https://github.com/huggingface/transformers) into `utils/`. And we have also modified some lines in the class `BertForMaskedLM` in `src/transformers/modeling_bert.py` while keep the other codes unchanged. 

You just need run 
```
pip install .
```
to install transformers manually.

### 2. More details
You can cd to `pretrain` or `finetune` to learn more details about pre-training or finetuning.







