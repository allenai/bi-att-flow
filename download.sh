#!/usr/bin/env bash

DATA_DIR=$HOME/data
mkdir $DATA_DIR

# Download SQuAD
SQUAD_DIR=$DATA_DIR/squad
mkdir $SQUAD_DIR
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O $SQUAD_DIR/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O $SQUAD_DIR/dev-v1.1.json


# Download CNN and DailyMail
# wget https://s3-us-west-2.amazonaws.com/minjoon/data/cnn/cnn.tgz -O $DATA_DIR/cnn.tgz
# wget https://s3-us-west-2.amazonaws.com/minjoon/data/dailymail/dailymail.tgz -O $DATA_DIR/dailymail.tgz


# Download GloVe
GLOVE_DIR=$DATA_DIR/glove
mkdir $GLOVE_DIR
wget http://nlp.stanford.edu/data/glove.6B.zip -O $GLOVE_DIR/glove.6B.zip
unzip $GLOVE_DIR/glove.6B.zip -d $GLOVE_DIR

# Download NLTK (for tokenizer)
# Make sure that nltk is installed!
python3 -m nltk.downloader -d $HOME/nltk_data punkt
