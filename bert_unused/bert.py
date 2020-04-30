# Credit to Denis Antyukhov
# https://towardsdatascience.com/building-a-search-engine-with-bert-and-tensorflow-c6fdc0186c8a
import tensorflow as tf
import os.path
import sys
import numpy as np
from bert_serving.server.graph import optimize_graph
from bert_serving.server.helper import get_args_parser
from bert_serving.server.bert.tokenization import FullTokenizer
from bert_serving.server.bert.extract_features import convert_lst_to_features

# Add Covid19_Search_Tool/src to python path
nb_dir = os.path.split(os.getcwd())[0]
data_dir = os.path.join(nb_dir,'src')
if data_dir not in sys.path:
    sys.path.append(data_dir)

from bert_utils import get_graph_output_file, get_estimator_spec, get_container_features_input_fn, get_vectorizer 

def get_bert_vectorizer():
    # input dir
    model_dir = 'uncased_L-12_H-768_A-12/' #@param {type:'string'}
    # output dir
    bert_data_dir = 'data/BERT/' #@param {type:'string'}
    graph_dir = bert_data_dir + 'graph/' #@param {type:'string'}
    # output filename
    graph_filename = 'extractor.pbtxt' #@param {type:'string'}
    pooling_strategy = 'REDUCE_MEAN' #@param ['REDUCE_MEAN', 'REDUCE_MAX', 'NONE']
    pooling_layer = '-2' #@param {type:'string'} 
    max_sequence_length = '64' #@param {type:'string'}
    graph_path = graph_dir + graph_filename #@param {type:'string'}
    vocab_path = model_dir + 'vocab.txt' #@param {type:'string'}
    input_names = ['input_ids', 'input_mask', 'input_type_ids']
    graph_output_pbtxt = get_graph_output_file(bert_data_dir, 
                                                graph_dir, 
                                                graph_filename, 
                                                model_dir, 
                                                max_sequence_length, 
                                                pooling_layer, 
                                                pooling_strategy)
    params = {
            'graph_path': graph_path,
            'vocab_path': vocab_path,
            'max_sequence_length': max_sequence_length,
            'input_names': input_names}
    estimator = tf.estimator.Estimator(model_fn = get_estimator_spec, 
                                        params = params) 
    bert_vectorizer = get_vectorizer(estimator, 
                                    get_container_features_input_fn, 
                                    params) 
    return bert_vectorizer

