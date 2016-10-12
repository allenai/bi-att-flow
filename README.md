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
Preprocess Stanford QA dataset (along with GloVe vectors) and save them in `data/squad`:
```python -m squad.prepro_simple```

## Training
