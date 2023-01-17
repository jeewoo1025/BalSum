# BalSum

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
To evaluate BalSum, please change `MODEL_PATH` on `run_evaluate.sh` and run below script:
```python
bash run_evaluate.sh
```
`MODEL_PATH` should be a subdirectory in the `./cache_cnndm` or `./cache_xsum`.