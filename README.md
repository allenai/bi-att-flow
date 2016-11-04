# Bi-directional Attention Flow for Machine Comprehension

Follow three simple steps.
For more advanced usages, see below.

## 0. Requirements
#### General
- Python (developed on 3.5.2)
- unzip

#### Python Packages
- tensorflow (deep learning library, verified on r0.11)
- nltk (NLP tools, verified on 3.2.1)
- tqdm (progress bar, verified on 4.7.4)
- jinja2 (for visaulization; if you only train and test, not needed)

## 1. Pre-processing
First, prepare data. Donwload SQuAD data and GloVe and nltk corpus
(~850 MB, this will download files to `$HOME/data`):
```
chmod +x download.sh; ./download.sh
```

Second, Preprocess Stanford QA dataset (along with GloVe vectors) and save them in `$PWD/data/squad` (~5 minutes):
```
python -m squad.prepro
```

## 2. Training
The model was trained with NVidia Titan X (Pascal Architecture, 2016).
The model requires at least 12GB of GPU RAM.
If your GPU RAM is smaller than 12GB, you can either decrease batch size (performance might degrade),
or you can use multi GPU (see below).
The training converges at 10~15k steps, and it took ~1.5s per step (i.e. ~6 hours).


To train:
```
python -m basic.cli --mode train --noload
```

## 3. Testing
To Test (~30 mins):
```
python -m basic.cli --mode test --batch_size 8 --eval_num_batches 0
```

This command loads the most recently saved model during training and begins testing on the test data.
Note that batch size is reduced to 8, because testing requires more memory per example.
After the process ends, it prints F1 and EM scores, and also outputs a json file (`$PWD/out/basic/00/answer/test-####.json`,
where `####` is the step # that the model was saved).
Note that the printed scores are not official (our scoring scheme is a bit harsher).
To obtain the official number, use the official evaluator and the output json file:

```
python squad/evaluate-v1.1.py $HOME/data/squad/dev-v1.1.json out/basic/00/answer/test-####.json
```


## Results
See [SQuAD Leaderboard][squad]


<!--
## Using Pre-trained Model

If you would like to use pre-trained model, it's very easy! 
You can download the model weights [here][save] (make sure that its commit id matches the source code's).
Extract them and put them in `$PWD/out/basic/00/save` directory, with names unchanged.
Then do the testing again, but you need to specify the step # that you are loading from:
```
python -m basic.cli --mode test --batch_size 8 --eval_num_batches 0 --load_step ####
```
-->


## Multi-GPU Training & Testing
Our model supports multi-GPU training.
We follow the parallelization paradigm described in [TensorFlow Tutorial][multi-gpu].
In short, if you want to use batch size of 60 (default) but if you have 3 GPUs with 4GB of RAM,
then you initialize each GPU with batch size of 20, and combine the gradients on CPU.
This can be easily done by running:
```
python -m basic.cli --mode train --noload --num_gpus 3 --batch_size 20
```

Similarly, you can speed up your testing by (if your GPU's RAM is 4GB, then batch size should be 2 or 3):
```
python -m basic.cli --mode test --batch_size 2 --num_gpus 3
```
 

[multi-gpu]: https://www.tensorflow.org/versions/r0.11/tutorials/deep_cnn/index.html#training-a-model-using-multiple-gpu-cards
[save]: #
[squad]: http://stanford-qa.com
