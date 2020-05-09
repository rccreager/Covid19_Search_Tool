from pathlib import Path, PurePath
import pandas as pd
import nltk
import os
import sys
import re 
from time import time

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from bert_serving.client import BertClient

# Add Covid19_Search_Tool/src to python path
nb_dir = os.path.split(os.getcwd())[0]
data_dir = os.path.join(nb_dir,'src')
if data_dir not in sys.path:
    sys.path.append(data_dir)

# Import local libraries
from utils import ResearchPapers
from nlp import SearchResults, WordTokenIndex, preprocess, get_preprocessed_abstract_text, print_top_words

preprocessed = get_preprocessed_abstract_text('data/CORD-19-research-challenge/', 'metadata.csv')

# TO-DO: add optional loading of existing CSV embedding files
with BertClient(ip = '1.2.4.8') as bc:
    print('encoding...if client hangs, make sure you can have set up and can connect to server')
    bert_vectors = bc.encode(preprocessed, is_tokenized=False)

lda_bert = LatentDirichletAllocation(n_components = 3, learning_offset = 50., verbose=2)
t0 = time()
lda_bert.fit(bert_vectors)
print("BERT: done in %0.3fs" % (time() - t0))
with open('uncased_L-12_H-768_A-12/vocab.txt') as vocab_file:
    bert_feature_names = vocab_file.read().splitlines()
print('Topics from LDA using BERT:')
print_top_words(lda_bert, bert_feature_names, 25)

