# Training an Error Correction Model
### Preparation
1. You should have the dataset that includes "feedback".

### Data Preprocessing
```
python preprocess.py --sep --strip --use_modified_schema --train [train data path] --dev [dev data path] --test [test data path] --target edits --format feqs --out_dir [out folder dir]
```
### Running the train.py
```
cd $ISP_HOME/error_correction
python train.py --data_dir [root folder of all processed data] --data_revision [name of processed data folder] --model t5-base
```
### Running the prediction.py
```
cd $ISP_HOME/error_correction
python train.py --data_dir [root folder of all processed data] --data_revision [name of processed data folder] --model t5-base --ckp [path of checkpoint]
```