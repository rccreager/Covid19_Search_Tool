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

class DataContainer:
  # class for holding text data
  def __init__(self):
    self._texts = None
 
  def set(self, texts):
    if type(texts) is str:
      texts = [texts]
    self._texts = texts
   
  def get(self):
    return self._texts

def get_graph_output_file(
                        bert_data_dir,
                        graph_dir,
                        graph_filename,
                        model_dir,
                        max_sequence_length,
                        pooling_layer,
                        pooling_strategy):
    if not os.path.exists(bert_data_dir):
        os.mkdir(bert_data_dir)
    if not os.path.exists(graph_dir):
        os.mkdir(graph_dir)

    tf.io.gfile.mkdir(graph_dir)
    parser = get_args_parser()
    carg = parser.parse_args(args=['-model_dir', model_dir,
                                   '-graph_tmp_dir', graph_dir,
                                   '-max_seq_len', str(max_sequence_length),
                                   '-pooling_layer', str(pooling_layer),
                                   '-pooling_strategy', pooling_strategy])
    tmp_name, config = optimize_graph(carg)
    graph_output_file = os.path.join(graph_dir, graph_filename)

    tf.gfile.Rename(
        tmp_name,
        graph_output_file,
        overwrite=True
    )
    print("\nSerialized graph to {}".format(graph_output_file))
    return graph_output_file

def get_feed_dict(texts, params):
    bert_tokenizer = FullTokenizer(params['vocab_path'])
    log = tf.get_logger()
    text_features = list(convert_lst_to_features(
        texts,
        int(params['max_sequence_length']),
        int(params['max_sequence_length']),
        bert_tokenizer,
        log))
    target_size = len(texts)
    target_shape = (target_size, -1)
    feed_dict = {}
    for input_name in params['input_names']:
        features_i = np.array([getattr(f, input_name) for f in text_features])
        if len(features_i) % target_size != 0:
            num_feature_sets = len(features_i) // target_size
            features_i = features_i[0: target_size * num_feature_sets - 1]
        features_i = features_i.reshape(target_shape).astype("int32")
        feed_dict[input_name] = features_i
    return feed_dict

def get_container_features_input_fn(container, params):
    def gen():
        while True:
          try:
            yield get_feed_dict(container.get(), params)
          except StopIteration:
            yield get_feed_dict(container.get(), params)
    def input_fn():
        return tf.data.Dataset.from_generator(
            gen,
            output_types={input_name: tf.int32 for input_name in params['input_names']},
            output_shapes={input_name: (None, None) for input_name in params['input_names']})
    return input_fn

def get_estimator_spec(features, mode, params):
    with tf.io.gfile.GFile(params['graph_path'], 'rb') as graph_file:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(graph_file.read())

    output = tf.import_graph_def(graph_def,
                                 input_map = {k + ':0': features[k]
                                                for k in params['input_names']},
                                 return_elements = ['final_encodes:0'])
    return tf.estimator.EstimatorSpec(mode=mode, predictions={'output': output[0]})

def get_estimator_spec(features, mode, params):
    with tf.io.gfile.GFile(params['graph_path'], 'rb') as graph_file:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(graph_file.read())

    output = tf.import_graph_def(graph_def,
                                 input_map = {k + ':0': features[k]
                                                for k in params['input_names']},
                                 return_elements = ['final_encodes:0'])
    return tf.estimator.EstimatorSpec(mode=mode, predictions={'output': output[0]})

def batch(iterable, n=1):
    l = len(iterable)
    print(f'iterate {l} over {n}')
    for index in range(0, l, n):
        yield iterable[index : min(index + n, l)]

def get_vectorizer(_estimator, _input_fn_builder, params, batch_size=128):
  container = DataContainer()
  predict_fn = _estimator.predict(_input_fn_builder(container, params),
                                yield_single_examples=False)
  def vectorize(text, verbose=False):
    x = []
    bar = tf.keras.utils.Progbar(len(text))
    for text_batch in batch(text, batch_size):
      container.set(text_batch)
      x.append(next(predict_fn)['output'])
      if verbose:
        bar.add(len(text_batch))
    r = np.vstack(x)
    return r
  return vectorize

