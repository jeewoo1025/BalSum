# BalSum: Balancing Lexical and Semantic Quality in Abstractive Summarization
This repository contains the code for our paper "[Balancing Lexical and Semantic Quality in Abstractive Summarization (ACL short, 2023)](https://aclanthology.org/2023.acl-short.56/)". 

## Overview
We propose a novel training method in which a re-ranker balances lexical and semantic quality. Based on a two-stage framework, our model, named **BalSum**, is trained on multi-task learning. We directly reflect the ROUGE score difference on a ranking loss to preserve the lexical quality as much as possible. Then, we use a contrastive loss with instance weighting to identify summaries whose meanings are close to the document. Specifically, we define novel false positives (semantic mistakes) and present a strategy to reduce their influence in ranking.

## How to Install
### Requirements
* `python3.8`

Run the following script to install the additional libraries
```python
pip install -r requirements.txt
```

### Description of Codes
We implement our model based on Huggingface [Transformers](https://github.com/huggingface/transformers) library.
* `cal_rouge.py`, `cal_bertscore.py` : ROUGE / BERTScore calculation
* `config.py` : model configuration
* `data_utils.py` : dataloader
* `model.py` : model architecture 
* `main.py` : training and evaluation procedure
* `utils.py` : utility functions

### Workspace
You should create following directories for our experiments: 
* `./cache_cnndm`, `./cache_xsum` : save model checkpoints for each dataset
* `./result` : save evaluation results

## Dataset
We experiment on two datasets.
* [CNN/DM](https://github.com/abisee/cnn-dailymail)
* [XSum](https://github.com/EdinburghNLP/XSum)

### Prepare Candidate Summaries
We referred to [BRIO](https://github.com/yixinL7/BRIO) code when we generated and preprocessed candidate summaries. 
Additionally, I measured cosine-similarity between the reference and each candidate summaries using [SimCSE model](https://github.com/princeton-nlp/SimCSE) for instance weighting strategy. Then, we classify them above each threshold (~0.9) and save them in the dateset file by the threshold.

## How to Run
### Training
You can change the specific settings in `config.py`. 
```python
python main.py --cuda --gpuid [list of gpuids] -l --config [(cnndm/xsum)] --wandb [Project Name of Wandb]
```

<b>Example: training on CNN/DM</b>
```python
python main.py --cuda --gpuid 0 -l --config cnndm --wandb CNNDM_train
```

### Evaluation
For ROUGE calculation, we use the standard [ROUGE Perl Package](https://github.com/summanlp/evaluation/tree/master/ROUGE-RELEASE-1.5.5). We lowercased and tokenized (using PTB Tokenizer) texts before calculating ROUGE scores. 
To evaluate BalSum, please change `MODEL_PATH` on `run_evaluate.sh` and run below script:
```python
bash run_evaluate.sh
```
`MODEL_PATH` should be a subdirectory in the `./cache_cnndm` or `./cache_xsum`.

## Citation
Please cite our paper if you use BalSum in your work:
```
@inproceedings{sul-choi-2023-balancing,
    title = "Balancing Lexical and Semantic Quality in Abstractive Summarization",
    author = "Sul, Jeewoo  and  Choi, Yong Suk",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-short.56",
    pages = "637--647",
    abstract = "An important problem of the sequence-to-sequence neural models widely used in abstractive summarization is exposure bias. To alleviate this problem, re-ranking systems have been applied in recent years. Despite some performance improvements, this approach remains underexplored. Previous works have mostly specified the rank through the ROUGE score and aligned candidate summaries, but there can be quite a large gap between the lexical overlap metric and semantic similarity. In this paper, we propose a novel training method in which a re-ranker balances the lexical and semantic quality. We further newly define false positives in ranking and present a strategy to reduce their influence. Experiments on the CNN/DailyMail and XSum datasets show that our method can estimate the meaning of summaries without seriously degrading the lexical aspect. More specifically, it achieves an 89.67 BERTScore on the CNN/DailyMail dataset, reaching new state-of-the-art performance. Our code is publicly available at https://github.com/jeewoo1025/BalSum.",
}
```

If you have any questions, please put them on github issue. Thank you for your interests.
