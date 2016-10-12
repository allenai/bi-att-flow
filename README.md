# Textbook QA

## Requirements
### General
- Python (developed on 3.5.2)
- unzip

### Python Packages
- tensorflow
- nltk
- progressbar2
- tqdm
- networkx
- requests

```
# install tensorflow referring to tensorflow.org
pip install nltk progressba2 tqdm networkx requests
```

## Preprocessing
Donwload SQuAD data and GloVE and nltk corpus:
```chmod +x download.sh; ./download.sh```

Preprocess Stanford QA dataset (along with GloVe vectors) and save them in `data/squad`:
```python -m squad.prepro_simple```

## Training
```CUDA_VISIBLE_DEVICES=0 python -m basic.cli --mode train --noload --attention --use_glove_for_unk --known_if_glove --nofinetune --notraditional --device /gpu:0```
