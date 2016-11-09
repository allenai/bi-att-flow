# Bi-directional Attention Flow for Machine Comprehension

This is the original implementation of the following paper:

[Bi-directional Attention Flow for Machine Comprehension][paper] (Seo et al., 2016) 

Regarding [Stanford Question Answering Dataset][squad] experiment, see Section A.
Regarding [CNN/DailyMail Dataset][cnn] experiments, see Section B.

## Common Requirements
### General
- Python (developed on 3.5.2)
- unzip

### Python Packages
- tensorflow (deep learning library, verified on r0.11)
- nltk (NLP tools, verified on 3.2.1)
- tqdm (progress bar, verified on 4.7.4)
- jinja2 (for visaulization; if you only train and test, not needed)

## Section A: Stanford Question Answering

### 1. Pre-processing
First, prepare data. Donwload SQuAD data and GloVe and nltk corpus
(~850 MB, this will download files to `$HOME/data`):
```
chmod +x download.sh; ./download.sh
```

Second, Preprocess Stanford QA dataset (along with GloVe vectors) and save them in `$PWD/data/squad` (~5 minutes):
```
python -m squad.prepro
```

### 2. Training
The model was trained with NVidia Titan X (Pascal Architecture, 2016).
The model requires at least 12GB of GPU RAM.
If your GPU RAM is smaller than 12GB, you can either decrease batch size (performance might degrade),
or you can use multi GPU (see below).
The training converges at 10~15k steps, and it took ~1.5s per step (i.e. ~6 hours).


To train:
```
python -m basic.cli --mode train --noload
```

### 3. Testing
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

### 4. Ensemble
Ensemble method gives even better performance.


### 5. Results
See [paper][paper] or [SQuAD Leaderboard][squad].


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


### 6. Multi-GPU Training & Testing
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

## Section B: CNN/DailyMail 

We show the procedure for CNN data. Similar steps can be taken for DailyMail.

### 1. Pre-processing

First, prepare data. Download the CNN Questions from [here][cho-cnn].
Extract it in a folder (e.g. `$HOME/data/cnn/`).

Then perform pre-processing script on the data:
```bash
python -m cnn_dm.prepro --source_dir $PATH_TO_CNN_DATA/questions --target_dir data/cnn/
```
This creates auxilary files in local `data/cnn` folder.
Note that this filters out some long questions or those with long articles (< 5% of questions) for reduced GPU RAM usage.
In order to test on the full data, we pre-process without filtering and put it in another folder:
```bash
python -m cnn_dm.prepro --source_dir $PATH_TO_CNN_DATA/questions --target_dir data/cnn-all/ --num_sents_th 9999 --ques_size_th 9999
```

### 2. Training
Due to very lengthy articles, single GPU of 12GB RAM can only take batch size of 15.
Since our model needs a batch size of 30, we use two GPUs to resolve this issue (see Section A.2 and A.6).
First make sure you have access to two GPUs (by setting CUDA_VISIBLE_DEVICES), and run:

```bash
python -m basic_cnn.cli --mode train --noload --root_dir $PATH_TO_CNN_DATA/questions --data_dir data/cnn/ --num_gpus 2
```

This will run for 2-3 days on Titan X, though you can early stop at your convenience (saved every 1000 step).
Monitor the dev accuracy and loss via Tensorboard.


### 3. Testing
As noted in A.1, testing must be done on entire test data without filtering:
```bash
python -m basic_cnn.cli --mode test --root_dir $PATH_TO_CNN_DATA\questions --data_dir data/cnn-all/ --num_gpus 2 --batch_size 5 --eval_num_batches 0
```
Make sure that the total number of questions by the system matches the true number of questions (so that no question is filtered out).  
Decrease batch size if you get OOM; the size doesn't have any effect on the accuracy.
You can use as many GPUs you want to test faster.


### 4. Results

See [paper][paper]

[multi-gpu]: https://www.tensorflow.org/versions/r0.11/tutorials/deep_cnn/index.html#training-a-model-using-multiple-gpu-cards
[save]: #
[squad]: http://stanford-qa.com
[paper]: https://arxiv.org/abs/1611.01603
[cnn]: https://github.com/deepmind/rc-data
[cho-cnn]: http://cs.nyu.edu/~kcho/DMQA/
