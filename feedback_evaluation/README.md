# Training a User Feedback Evaluator
### Set up the virtual environment

1. Follow all instructions on the [project homepage](https://github.com/hyan5/Learning_to_Simulate_NL_Feedback/tree/main) to set up the python virtual environment and install all dependencies.

2. Download all necessary datasets and put them into the "$ISP_HOME/data" folder.

3. Generate the template feedback for SPLASH data and remove structural errors

### Running the train.py
```
cd $ISP_HOME/feedback_evaluation
python train.py --tables [spider table file] --train [train file] --dev [dev file]

optional arguments:
	--model	embedding model to fine-tune, currently support "roberta-large"
	--lr	learning rate for fine-tuning
	--margin	expected margin between positive and negative score
	--prior_loss	the strength of prior term in loss calculation
	--epoch	the total number of epochs for training
	--batch_size	the number of examples in one batch
	--negative_mode	how to generate negative examples, currently support "replace"
	--negative_num	the maximum number of negative examples can be generated for each input
```