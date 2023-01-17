#!/bin/bash

# setting
MODEL_PATH='[Your directory]'
CONFIG='cnndm'

python main.py --cuda --gpuid 0 -e --config $CONFIG --model_pt $MODEL_PATH/best_model.bin

# ROUGE
python cal_rouge.py --ref ./result/$MODEL_PATH/reference --hyp ./result/$MODEL_PATH/candidate 

# BERTScore
python cal_bertscore.py --ref ./result/$MODEL_PATH/reference --hyp ./result/$MODEL_PATH/candidate 