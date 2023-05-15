# Utility Functions

## 1. Template Feedback Generation

### Method 1: using the command line tool `generate_template_feedback.py`

``` shell scripts
python generate_template_feedback.py -i input.json -o output.json
```

#### Inputs

The `input.json` should be a text file in JSON format. For each item of it, For each item of it, "db_id", "gold_parse", and "predicted_parse_with_values" terms must exist. The template-based feedback which correct "predicted_parse_with_values" into "gold_parse" will be added to the corresponding item as a new term "generated_feedback". The span ranges of "primary" and "secondary" will also be calculated. Then the original data, with feedback and span information added will be write to `output.json`

Templates are flexible, there are several options to select while generating feedback:

* `--no_underscore` removes all the underscores.
* `--no_quote` removes quotation marks around values. If this option is not specified, the string value in SQL will be with quotation marks.
* `--show_tag` adding "add", "sub", and "info" tags to the corresponding part of the feedback.
* `--connect_foreign_key_group` When comparing two column names, not only consider the column name itself. Because "airport.id" and "flight.id" are different things although they have identical clumn name "id". But we can't consider both the table name and column name, because "list.lastname" and "teacher.lastname" could be the same thing. Thus, we should consider the foreign key relationship between two columns. If they are foreign keys of each other, they are considered as the same thing. In addition, if A is B's foreign key, C is B's foreign key, C should also be equal to A. If this option is specified, the foreign key relationship is considered, otherwise we will compare both the table name and column name.
* `--only_column_name` If this option is specified, we compare the column name solely when judging the equivelance between two columns.
* `--no_nltk_tokenizer` If this option is specified, the feedback will not use NLTK tokenizer to tokenize them. Thus, the punctuations are not separated with the previous workd by a space, like **"This is a sentence."**. By default, punctuations are separated like **"This is a sentence ."**.

Note: If neither the `--connect_foreign_key_group` and `--only_column_name` are specified, both of the column name and table name will be compared while judging the equivelent between two column names.

#### Output

This command will generate two files, one is `output.json` which stores all successfully processed data with `template_feedback`, `predicted_parse_explanation`, `primary_span`, and `secondary_span` fields added to each sample.

The another one is `output_struct_err.json` which stores all samples with mistakes with `error info` added to each sample. There are three kinds of mistakes:
1. `StructureError` represents a defined structural error between the predicted parse and the golden parse.
2. `Exception` represents an undefined error occured. There are no `Exception` in the SPLASH dataset.
3. Empty string represents that the parser deem the predicted parse and the golden parse are the same so there is no feedback.

### Method 2: Use the function `util.template.feedback`
 
``` python
from util.template import feedback
import config

config.REPLACE_UNDERSCORE_WITH_SPACE = True
config.VALUE_KEEP_QUOTE = False
config.SHOW_TAG = True
config.CONNECT_FOREIGN_KEY_GROUP = True

db_id = "allergy_1"
gold_sql = "SELECT count(*) FROM Student WHERE sex  =  \"M\" AND StuID IN (SELECT StuID FROM Has_allergy AS T1 JOIN Allergy_Type AS T2 ON T1.Allergy  =  T2.Allergy WHERE T2.allergytype  =  \"food\")"

pred_sql = "SELECT Count ( * ) FROM Allergy_Type AS T1 JOIN Has_Allergy AS T2 ON T1.Allergy = T2.Allergy JOIN Student AS T3 ON T2.StuID = T3.StuID WHERE T3.Sex = \"M\" AND T1.AllergyType = \"food\""

fb = feedback(pred_sql, gold_sql, db_id)
```
The `config` values corresponds to the command line tool options.

---
## 2. Template Explanation Generation

### Method 1: using the command line tool `generate_explanation.py`

``` shell scripts
python generate_explanation.py -i input.json -o output.json --no_nltk_tokenizer
```

Please not that the `input.json` is a text file in JSON format. The "predicted_parse_with_values" term is required. The program will generate template-based explanation then added as a new term "generated_explanation".

* `--no_nltk_tokenizer` If this option is specified, the explanation will not use NLTK tokenizer. Thus, the punctuations are not separated with the previous workd by a space, like **"This is a sentence."**. By default, punctuations are separated like **"This is a sentence ."**.

### Method 2: import the function `templates.explanation`

``` python
from utils.templates import explanation

db_id = "allergy_1"

pred_sql = "SELECT Count ( * ) FROM Allergy_Type AS T1 JOIN Has_Allergy AS T2 ON T1.Allergy = T2.Allergy JOIN Student AS T3 ON T2.StuID = T3.StuID WHERE T3.Sex = \"M\" AND T1.AllergyType = \"food\""

exp = explanation(pred_sql, db_id)
```

---
## 3. Edit Distance Calculation

SQL-Edits denotes the changes needed to transform a source query to a target query. For e.g., 
```
SOURCE: 'SELECT first_name, last_name FROM players ORDER BY birth_date ASC' 

TARGET: 'SELECT first_name, last_name FROM players WHERE hand = "left" and last_name = "ram" ORDER BY birth_date ASC' 

SQL-Edits:
1) <where> add col:players.hand equals </where> 
2) <where> add col:players.last_name equals </where> 
i.e., missing a where clause with column "hand" and "last_name" 

Total edit distance: 2 
```
 The implementation follows the same spirit of [the NL-Edit paper](https://arxiv.org/pdf/2103.14540.pdf). Please follow [here to read more about the clauses and the arguments](readme_clause.md).


We have implemented two methods to calculate the edits and the edit distance.

### Method 1: using the command line tool `calculate_editsize.py`  

``` shell scripts
python calculate_editsize.py -i input.json -o output.json
```

The `input.json` should be a text file in JSON format. For each item of it, "db_id", "gold_parse", and "predicted_parse_with_values" terms must exist. EditSize is calculated between the "gold_parse" and the "predicted_parse_with_values". After the calculation, an extra "editsize" term will be added to the corresponding item and the output will be written to the `output.json`.

Notes: the statistics of the EditSize of input data will be printed out to the shell.

### Method 2: import the function `utils.edit_size`

``` python
from utils.utils_ import edit_size

db_id = "allergy_1"
gold_sql = "SELECT count(*) FROM Student WHERE sex  =  \"M\" AND StuID IN (SELECT StuID FROM Has_allergy AS T1 JOIN Allergy_Type AS T2 ON T1.Allergy  =  T2.Allergy WHERE T2.allergytype  =  \"food\")"

pred_sql = "SELECT Count ( * ) FROM Allergy_Type AS T1 JOIN Has_Allergy AS T2 ON T1.Allergy = T2.Allergy JOIN Student AS T3 ON T2.StuID = T3.StuID WHERE T3.Sex = \"M\" AND T1.AllergyType = \"food\""
size = edit_size(pred_sql, gold_sql, db_id)
```
---