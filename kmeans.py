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

from bert_serving.server.graph import optimize_graph
from bert_serving.server.helper import get_args_parser
from bert_serving.server.bert.tokenization import FullTokenizer
from bert_serving.server.bert.extract_features import convert_lst_to_features

# Add Covid19_Search_Tool/src to python path
nb_dir = os.path.split(os.getcwd())[0]
data_dir = os.path.join(nb_dir,'src')
if data_dir not in sys.path:
    sys.path.append(data_dir)

# Import local libraries
from utils import ResearchPapers
from nlp import SearchResults, WordTokenIndex, preprocess

# base model dir
nb_dir = os.path.split(os.getcwd())[0]
base_dir = os.path.join(nb_dir,'models')

# input dir
MODEL_DIR = 'uncased_L-12_H-768_A-12/' #@param {type:"string"}
# output dir
if not os.path.exists('data/BERT/graph/'):
    os.mkdir('data/BERT/')
    os.mkdir('data/BERT/graph/')
GRAPH_DIR = 'data/BERT/graph/' #@param {type:"string"}
# output filename
GRAPH_OUT = 'extractor.pbtxt' #@param {type:"string"}

POOL_STRAT = 'REDUCE_MEAN' #@param ['REDUCE_MEAN', 'REDUCE_MAX', "NONE"]
POOL_LAYER = '-2' #@param {type:"string"}
SEQ_LEN = '256' #@param {type:"string"}

tf.gfile.MkDir(GRAPH_DIR)

parser = get_args_parser()
carg = parser.parse_args(args=['-model_dir', MODEL_DIR,
                               '-graph_tmp_dir', GRAPH_DIR,
                               '-max_seq_len', str(SEQ_LEN),
                               '-pooling_layer', str(POOL_LAYER),
                               '-pooling_strategy', POOL_STRAT])

tmp_name, config = optimize_graph(carg)
graph_fout = os.path.join(GRAPH_DIR, GRAPH_OUT)

tf.gfile.Rename(
    tmp_name,
    graph_fout,
    overwrite=True
)
print("\nSerialized graph to {}".format(graph_fout))

log = logging.getLogger('tensorflow')
log.setLevel(logging.INFO)
log.handlers = []

GRAPH_PATH = GRAPH_DIR + GRAPH_OUT #@param {type:"string"}
VOCAB_PATH = MODEL_DIR + "vocab.txt" #@param {type:"string"}

SEQ_LEN = 256 #@param {type:"integer"}

INPUT_NAMES = ['input_ids', 'input_mask', 'input_type_ids']
bert_tokenizer = FullTokenizer(VOCAB_PATH)

def build_feed_dict(texts):
    
    text_features = list(convert_lst_to_features(
        texts, SEQ_LEN, SEQ_LEN, 
        bert_tokenizer, log, False, False))

    target_shape = (len(texts), -1)

    feed_dict = {}
    for iname in INPUT_NAMES:
        features_i = np.array([getattr(f, iname) for f in text_features])
        features_i = features_i.reshape(target_shape).astype("int32")
        feed_dict[iname] = features_i

    return feed_dict
