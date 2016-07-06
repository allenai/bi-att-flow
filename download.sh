#!/usr/bin/env bash

DATA_DIR=$HOME/data
mkdir $DATA_DIR

# Download SQuAD
SQUAD_DIR=$DATA_DIR/squad
mkdir $SQUAD_DIR
wget https://stanford-qa.com/train-v1.0.json -O $SQUAD_DIR/train-v1.0.json
wget https://stanford-qa.com/dev-v1.0.json -O $SQUAD_DIR/dev-v1.0.json


# Download GloVe

# Download NLTK (for tokenizer)
# Make sure that nltk is installed!
python -m nltk.downloader -d $HOME/nltk_data punkt
