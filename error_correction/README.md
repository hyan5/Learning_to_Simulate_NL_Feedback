# Training an Error Correction Model
### Preparation
You should first combine different train files into one file (e.g. full SPLASH + EditSQL w/ simulated feedback, 20% SPLASH + 80% SPLASH w/ simulated feedback, etc)

### Data Preprocessing
```
python preprocess.py --sep --strip --use_modified_schema --train [train data path] --dev [dev data path] --test [test data path] --target edits --format feqs --out_dir [out folder dir]
```
### Running  train.py
```
cd $ISP_HOME/error_correction
python train.py --data_dir [root folder of all processed data] --data_revision [name of processed data folder] --model t5-base
```
### Running  prediction.py
```
cd $ISP_HOME/error_correction
python train.py --data_dir [root folder of all processed data] --data_revision [name of processed data folder] --model t5-base --ckp [path of checkpoint]
```
### Running  train.py w/ full SPLASH + EditSQL w/ simulated feedback, then test on SPLASH/EditSQL test set
Combine full SPLASH and EditSQL train w/ simulated feedback into one file "splash_editsql_simulated_feedback.json".
```
python preprocess.py --sep --strip --use_modified_schema --train ../data/splash/split/splash_editsql_simulated_feedback.json --dev ../data/splash/dev_w_template_feedback.json --test ../data/splash/test_w_template_feedback.json --target edits --format feqs --out_dir data/splash_editsql_feqs_edits
python train.py --data_dir data --data_revision splash_editsql_feqs_edits --model t5-base
```
### Running  train.py w/ 20% SPLASH + 80% SPLASH w/ simulated feedback, then test on SPLASH test set
Combine full SPLASH and EditSQL train w/ simulated feedback into one file "splash_20_80_w_simulated_feedback.json".
```
python preprocess.py --sep --strip --use_modified_schema --train ../data/splash/split/splash_20_80_w_simulated_feedback.json --dev ../data/splash/dev_w_template_feedback.json --test ../data/splash/test_w_template_feedback.json --target edits --format feqs --out_dir data/splash_20_80_feqs_edits
python train.py --data_dir data --data_revision splash_20_80_feqs_edits --model t5-base
```
### Running  train.py w/ 10% SPLASH + 90% SPLASH w/ simulated feedback, then test on SPLASH test set
Combine full SPLASH and EditSQL train w/ simulated feedback into one file "splash_10_90_w_simulated_feedback.json".
```
python preprocess.py --sep --strip --use_modified_schema --train ../data/splash/split/splash_10_90_w_simulated_feedback.json --dev ../data/splash/dev_w_template_feedback.json --test ../data/splash/test_w_template_feedback.json --target edits --format feqs --out_dir data/splash_10_90_feqs_edits
python train.py --data_dir data --data_revision splash_10_90_feqs_edits --model t5-base
```
### Running  train.py w/ 5% SPLASH + 95% SPLASH w/ simulated feedback, then test on SPLASH test set
Combine full SPLASH and EditSQL train w/ simulated feedback into one file "splash_5_95_w_simulated_feedback.json".
```
python preprocess.py --sep --strip --use_modified_schema --train ../data/splash/split/splash_5_95_w_simulated_feedback.json --dev ../data/splash/dev_w_template_feedback.json --test ../data/splash/test_w_template_feedback.json --target edits --format feqs --out_dir data/splash_5_95_feqs_edits
python train.py --data_dir data --data_revision splash_5_95_feqs_edits --model t5-base
```