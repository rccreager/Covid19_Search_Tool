# scientific and numberical libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#general libraries
from pathlib import Path, PurePath
import requests
from requests.exceptions import HTTPError, ConnectionError
import re, os, sys
import logging

# NLP libraries
import nltk

# Import Tensorflow libraries 
import tensorflow as tf
from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.estimator.model_fn import EstimatorSpec
from tensorflow.keras.utils import Progbar

# Add Covid19_Search_Tool/src to python path
nb_dir = os.path.split(os.getcwd())[0]
src_dir = os.path.join(nb_dir,'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Import local libraries
from utils import ResearchPapers
from nlp import SearchResults, WordTokenIndex, preprocess
from bert import get_bert_vectorizer

bert_vectorizer, bert_feature_names = get_bert_vectorizer()
