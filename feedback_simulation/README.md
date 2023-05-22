# Training a User Feedback Simulator

### Set up the virtual environment

1. Follow all instructions on the [project homepage](https://github.com/hyan5/Learning_to_Simulate_NL_Feedback/tree/main) to set up the python virtual environment and install all dependencies.

2. Download all necessary datasets and put them into the "$ISP_HOME/data" folder.

3. Generate the template feedback for SPLASH data and remove structural errors.

4. If reproduce our experiments, you should download Collected error prases on EditSQL and repeat the previous step to remove structural errors in EditSQL:
```
cd $ISP_HOME/utils
python generate_template_feedback.py -i ../data/editsql/train.json -o ../data/editsql/train_w_template_feedback.json --no_underscore --no_quote --connect_foreign_key_group --use_modified_schema
python generate_template_feedback.py -i ../data/editsql/dev.json -o ../data/editsql/dev_w_template_feedback.json --no_underscore --no_quote --connect_foreign_key_group --use_modified_schema
python generate_template_feedback.py -i ../data/editsql/test.json -o ../data/editsql/test_w_template_feedback.json --no_underscore --no_quote --connect_foreign_key_group --use_modified_schema
```
### Data Preprocessing
```
cd $ISP_HOME/feedback_simulation
python preprocess.py --sep --strip --use_modified_schema --train [train data path] --dev [dev data path] --test [test data path] --target feedback --format tqes --out_dir [out dir]

arguments:
--sep			whether use the special tokens to separate the different parts
--strip		remove all white space at the end
--use_modified_schema		use the canonical name of database schema
--train  	the path of train file, e.g. "../data/splash/train_20_80.json"
--dev			the path of dev file, e.g. "../data/splash/dev_w_template_feedback.json"
--test		the path of test file, e.g. "../data/splash/test_w_template_feedback.json"
--target	the target output. use "edits" for error correction model
--format	the order to concatenate the inputs. (Use "tqes" to reproduce our experiments)
--out_dir where to store the processed data. 
```

### Running train.py

Put the evaluation checkpoint under the folder "eval_ckp".

```
python train.py --data_dir [root folder of all processed data] --data_revision [name of processed data folder] --model t5-large --evaluation_ckp [evaluator path]

arguments:
	--data_dir	# the root folder of all processed data
	--data_revision	# the name of processed data folder
	--model 	# the model type [t5-base, t5-large]
	--evaluation_ckp #feedback evaluation model path
```
### Running prediction.py
If you want to  load the checkpoint and run prediction:
```
python prediction.py --data_dir [root folder of all processed data] --data_revision [name of processed data folder] --model t5-large --ckp [path of checkpoint] --evaluation_ckp [evaluator path]

arguments:
	--data_dir	# the root folder of all processed data
	--data_revision	# the name of processed data folder
	--model 	# the model type [t5-base, t5-large]
	--ckp #the path of saved checkpoint
	--evaluation_ckp #feedback evaluation model path
```
### Running train.py w/ full SPLASH, then simulate feedback on EditSQL
Put feedback evaluation model trained w/ full SPLASH under folder "eval_ckp"
```
python preprocess.py --sep --strip --use_modified_schema --train ../data/splash/train_w_template_feedback.json --dev dev_w_template_feedback.json --test ../data/editsql/test_w_template_feedback.json --target feedback --format tqes --out_dir data/editsql_tqes_feedback
python train.py --data_dir data --data_revision editsql_tqes_feedback --model t5-large --evaluation_ckp eval_ckp/evaluation_full_splash.pt
```
### Running train.py w/ 20% SPLASH, then simulate feedback on remaining 80% SPLASH
Put feedback evaluation model trained w/ 20% SPLASH under folder "eval_ckp"
```
python preprocess.py --sep --strip --use_modified_schema --train ../data/splash/split/train_20.json --dev dev_w_template_feedback.json --test ../data/splash/split/train_80.json --target feedback --format tqes --out_dir data/splash_20_80_tqes_feedback
python train.py --data_dir data --data_revision splash_20_80_tqes_feedback --model t5-large --evaluation_ckp eval_ckp/evaluation_20_splash.pt
```
### Running train.py w/ 10% SPLASH, then simulate feedback on remaining 90% SPLASH
Put feedback evaluation model trained w/ 10% SPLASH under folder "eval_ckp"
```
python preprocess.py --sep --strip --use_modified_schema --train ../data/splash/split/train_10.json --dev dev_w_template_feedback.json --test ../data/splash/split/train_90.json --target feedback --format tqes --out_dir data/splash_10_90_tqes_feedback
python train.py --data_dir data --data_revision splash_10_90_tqes_feedback --model t5-large --evaluation_ckp eval_ckp/evaluation_10_splash.pt
```
### Running train.py w/ 5% SPLASH, then simulate feedback on remaining 95% SPLASH
Put feedback evaluation model trained w/ 5% SPLASH under folder "eval_ckp"
```
python preprocess.py --sep --strip --use_modified_schema --train ../data/splash/split/train_5.json --dev dev_w_template_feedback.json --test ../data/splash/split/train_95.json --target feedback --format tqes --out_dir data/splash_5_95_tqes_feedback
python train.py --data_dir data --data_revision splash_5_95_tqes_feedback --model t5-large --evaluation_ckp eval_ckp/evaluation_5_splash.pt
```