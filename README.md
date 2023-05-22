# Learning to Simulate Natural Language Feedback for Interactive Semantic Parsing
This repository provides code implementation for our paper [Learning to Simulate Natural Language Feedback for Interactive Semantic Parsing](https://github.com/hyan5/Learning_to_Simulate_NL_Feedback.git) accepted by *ACL 2023*.

## 1. Overview
<p align="center">
<img src="overview.png" alt="Arch Overview" title="Overview" width="600"/>
</p>
Interactive semantic parsing based on natural language (NL) feedback, where users provide feedback to correct the parser mistakes, has emerged as a more practical scenario than the traditional one-shot semantic parsing. However, prior work has heavily relied on human-annotated feedback data to train the interactive semantic parser, which is prohibitively expensive and not scalable. In this work, we propose a new task of simulating NL feedback for interactive semantic parsing. We accompany the task with a novel feedback evaluator. The evaluator is specifically designed to assess the quality of the simulated feedback, based on which we decide the best feedback simulator from our proposed variants. On a text-to-SQL dataset, we show that our feedback simulator can generate high-quality NL feedback to boost the error correction ability of a specific parser. In low-data settings, our feedback simulator can help achieve comparable error correction performance as trained using the costly, full set of human annotations.

### 1.1 Simulating feedback to a specific semantic parse
We investigate whether our feedback simulator trained on the [SPLASH](https://aclanthology.org/2020.acl-main.187.pdf) dataset can simulate feedback for an unseen semantic parser. We first follow a similar procedure of SPLASH to create mistakes made by [EditSQL](https://arxiv.org/abs/1909.00786) on the Spider training set, and then apply our feedback simulator to simulate NL feedback. This results in around 2,400 simulated training examples. This data is then used to augment the original SPLASH training set for training an error correction model. We evaluate the error correction model on both the SPLASH test set and the EditSQL test set (which similarly contains human-annotated feedback to EditSQL’s mistakes on the Spider dev set and was additionally provided by [NL-Edit](https://arxiv.org/pdf/2103.14540.pdf).

### 1.2 Simulating feedback in low-data settings
One important motivation of our research is to reduce the need for human annotations. Therefore, we also experiment with a “low data” setting, where only *K%* of the SPLASH training set will be used to construct our feedback simulator and evaluator. For the remaining *(100−K)%* of training examples, we will instead apply our feedback simulator to simulate NL feedback. In experiments, we consider *K=20, 10, and 5*,, consuming 1639, 836, and 268 training examples, respectively.
## 2. Setup
This project is tested in python 3.8.6.

First, clone the repository and set up the `${ISP_HOME}` environment:
```
git clone git@github.com:hyan5/Learning_to_Simulate_NL_Feedback.git
export ISP_HOME=$(pwd)
export PYTHONPATH=$ISP_HOME:$ISP_HOME/utils:$PYTHONPATH
```

Then download the Spider data from [its official website](https://yale-lily.github.io/spider) and save it under the `data/spider/` folder. The data paths have been added to the global config file `config.py`.

Create a virtual environment and install all dependencies:
```
python -m venv ispenv 
source ispenv/bin/activate
pip install -r requirements.txt
```

## 2.1 Preparing SPLASH 
The first part of the project will prepare the SPLASH training/dev/test data, including removing structural errors and generating the template-based feedback for each instance. The generation of template-based explanation is skipped since the explanation has been provided by the original SPLASH datasets.

First, download the SPLASH data from [its official repository](https://github.com/MSR-LIT/Splash), and save them under `data/splash/`:
```
mkdir -p data/splash/
cd data/splash
wget https://raw.githubusercontent.com/MSR-LIT/Splash/master/data/train.json
wget https://raw.githubusercontent.com/MSR-LIT/Splash/master/data/dev.json
wget https://raw.githubusercontent.com/MSR-LIT/Splash/master/data/test.json
```

Then, run the following commands to process the SPLASH training data:
```
cd $ISP_HOME/utils
python generate_template_feedback.py -i ../data/splash/train.json -o ../data/splash/train_w_template_feedback.json --no_underscore --no_quote --connect_foreign_key_group --use_modified_schema
python generate_template_feedback.py -i ../data/splash/dev.json -o ../data/splash/dev_w_template_feedback.json --no_underscore --no_quote --connect_foreign_key_group --use_modified_schema
python generate_template_feedback.py -i ../data/splash/test.json -o ../data/splash/test_w_template_feedback.json --no_underscore --no_quote --connect_foreign_key_group --use_modified_schema
```
For details about options of `generate_template_feedback.py`, please refer to [the utility function README](utils/).

Now, you should have three files -- `train_w_template_feedback.json`, `dev_w_template_feedback.json`, and `test_w_template_feedback.json` -- under the `data/splash/` folder, all with template-based feedback. We will use these datasets to train a user feedback simulator.

## 2.2 Training a User Feedback Evaluator
Please refer to [Feedback Evaluation README](feedback_evaluation/)
## 2.3 Training a User Feedback Simulator
Please refer to [Feedback Simulation README](feedback_simulation/)

## 2.4 Training an Error Correction model

Please refer to [Error Correction README](error_correction/)

## 2.5 Experiments Reproduction

We uploaded all our data and checkpoints used in the experiments:
1. Collected error parses on EditSQL. :white_check_mark:

2. Our SPLASH data split in low-data settings:

   - 20-80 split :white_check_mark:

   - 10-90 split :white_check_mark:

   - 5-95 split :white_check_mark:

3. All checkpoints ([download]()):
   - Feedback Evaluation Models:
     - Trained with Full SPLASH data :white_check_mark:
     - Trained with 5/10/20 % SPLASH data :x: / :x:/ :white_check_mark:
   - Feedback Simulation Models:
     - Trained with Full SPLASH data :white_check_mark:
     - Trained with 5/10/20 % SPLASH data :x: / :x:/ :white_check_mark:
   - Error Correction Models:
     - Trained with SPLASH + EditSQL :white_check_mark:
     - Trained with *k%* SPLASH + *(100-k) %* SPLASH w/ simulated feedback :x: / :x:/ :white_check_mark:
   
   
