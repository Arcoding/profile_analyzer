# Import libraries
import numpy as np
import pandas as pd
import sys  
import os
import fitz
from unidecode import unidecode 
import re
import datetime
import shutil
import logging
from process_cv import process_one_cv
logging.basicConfig(filename='std.log',level=logging.DEBUG, filemode='w',format='%(asctime)s %(message)s',  datefmt='%m/%d/%Y %I:%M:%S %p',)
import spacy
import warnings
warnings.filterwarnings('ignore')




## Load NLP model (pre-trained)
nlp = spacy.load("en_core_web_md",disable=["ner"])

## Load skills json
skill_pattern_path = "../data/jz_skill_patterns.jsonl"

## Add entity ruler for skills.
## Added before NER to try to get this skills before any other entity predefined
ruler = nlp.add_pipe("entity_ruler", before='ner')
ruler.from_disk(skill_pattern_path)

## Add other customized pattern we wanna find
patterns = [{"label":'EMAIL',"pattern":[{"TEXT":{"REGEX":"([^@|\s]+@[^@]+\.[^@|\s]+)"}}]}]
ruler.add_patterns(patterns)


## Define folder with CVs
cv_folder = '../data/cv'
results ={'fname':[], 'n_columns':[]}

if len(sys.argv)>1:
    list_documents = [sys.argv[1]]
else:
    list_documents = [x for x in os.listdir(cv_folder) if x[0]!='.']

output_parent_folder = '../data/output/'
if not os.path.exists(output_parent_folder):
    os.mkdir(output_parent_folder)

## Start processing each CV
for fname in list_documents:
    print('*'*50)
    print(f'Processing {fname}')
    logging.debug(f'Processing {fname}')

    json_results = process_one_cv(fname,cv_folder,results,nlp )