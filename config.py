import os
from os.path import join
from typing import Counter

HOME = os.environ['ISP_HOME']

######################
## DATA PATH CONFIG ##
######################

# data path configuration
SPLASH_DIR = join(HOME, 'data', 'splash')   # SPLASH dataset
SPIDER_DIR = join(HOME, 'data', 'spider') # Spider dataset

SPLASH_TRAIN = join(SPLASH_DIR, 'train_w_template_feedback.json')
SPLASH_DEV = join(SPLASH_DIR, 'dev_w_template_feedback.json')
SPLASH_TEST = join(SPLASH_DIR, 'test_w_template_feedback.json')

SPIDER_TABLES = join(SPIDER_DIR, 'tables.json')
SPIDER_DBS = join(SPIDER_DIR, 'database')

# simulated data path configuration
SIM_DIR = join(HOME, 'simulated_data')
if not os.path.isdir(SIM_DIR):
    os.makedirs(SIM_DIR)


##############################
## TEMPLATE FEEDBACK CONFIG ##
##############################

# Configurations for Edit Parser and feedback generator
KEEP_TABLE_NAME = False
USE_MODIFIED_SCHEMA = True
REPLACE_UNDERSCORE_WITH_SPACE = False
KEEP_VALUE = True
VALUE_KEEP_QUOTE = True
SHOW_TAG = False
NO_NLTK_TOKENIZER = False
LOWER_SCHEMA_NAME = True

# Feedback Generation Templates
ABANDON = True      # report error while False rule encountered
ABANDONED = Counter()   # report the abbondoned samples by rules
RULES = [True for _ in range(100)] # TODO: what are these? Why 100?
RULES[1] = False    # disable "remove a sql operation"
RULES[3] = False    # disable "add a sql operation"
RULES[6] = False    # disable "remove a sql in FROM"
RULES[7] = False    # disable "add a sql in FROM"
RULES[20] = False   # disable "add complex sub-query in WHERE"
RULES[21] = False   # disable "remove sub-query in WHERE"

# Tags in feedback temeplates
ADD_TAG1 = '<primary>'
ADD_TAG2 = '</primary>'
SUB_TAG1 = '<secondary>'
SUB_TAG2 = '</secondary>'
INFO_TAG1 = ''
INFO_TAG2 = ''

# Whether to connect foreign key groups by JOIN ON condition # TODO: what are they? clean up?
CONNECT_FOREIGN_KEY_GROUP = False
CONNECT_FOREIGN_KEY_GROUP_BY_JOIN_ON = True
ONLY_COLUMN_NAME = False
