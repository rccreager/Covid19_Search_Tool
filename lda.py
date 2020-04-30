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

# Add Covid19_Search_Tool/src to python path
nb_dir = os.path.split(os.getcwd())[0]
data_dir = os.path.join(nb_dir,'src')
if data_dir not in sys.path:
    sys.path.append(data_dir)

# Import local libraries
from utils import ResearchPapers
from nlp import SearchResults, WordTokenIndex, preprocess, get_preprocessed_abstract_text, print_top_words
from bert import get_bert_vectorizer
from bert_utils import get_feed_dict

preprocessed = get_preprocessed_abstract_text('data/CORD-19-research-challenge/', 'metadata.csv', 128)
english_stopwords = list(set(stopwords.words('english')))

#tf_vectorizer = CountVectorizer(min_df=3, max_df=0.1, stop_words=english_stopwords)
#tf_vectors = tf_vectorizer.fit_transform(preprocessed)
#lda_tf = LatentDirichletAllocation(n_components = 3, learning_offset = 50., verbose=2)
#t0 = time()
#lda_tf.fit(tf_vectors)
#print("TF: done in %0.3fs" % (time() - t0))
#tf_feature_names = tf_vectorizer.get_feature_names()
#print('Topics from LDA using term frequency (TF):')
#print_top_words(lda_tf, tf_feature_names, 25)

#tfidf_vectorizer = TfidfVectorizer()
#tfidf_vectors = tfidf_vectorizer.fit_transform(preprocessed)
#lda_tfidf = LatentDirichletAllocation(n_components = 3, learning_offset = 50., verbose=2)
#t0 = time()
#lda_tfidf.fit(tfidf_vectors)
#print("TFIDF: done in %0.3fs" % (time() - t0))
#tfidf_feature_names = tfidf_vectorizer.get_feature_names()
#print('Topics from LDA using TF-IDF:')
#print_top_words(lda_tfidf, tfidf_feature_names, 25)

#bert_vectorizer = get_bert_vectorizer() 
#bert_vectors = bert_vectorizer(preprocessed)
batch_size = 256
num_parallel_calls = 2
num_clients = num_parallel_calls * 2  # should be at least greater than `num_parallel_calls`
# start a pool of clients
bc_clients = [BertClient(port=5555, show_server_config=False) for _ in range(num_clients)]
def get_encodes(x):
    # x is `batch_size` of lines, each of which is a json object
    samples = [json.loads(l) for l in x]
    text = [s['raw_text'] for s in samples]  # List[List[str]]
    labels = [s['label'] for s in samples]  # List[str]
    # get a client from available clients
    bc_client = bc_clients.pop()
    features = bc_client.encode(text, is_tokenized=True)
    # after use, put it back
    bc_clients.append(bc_client)
    return features, labels
bert_vectors = (tf.data.TextLineDataset(preprocessed).batch(batch_size)
        .map(lambda x: tf.py_func(get_encodes, [x], [tf.float32, tf.string]),  num_parallel_calls=num_parallel_calls)
        .map(lambda x, y: {'feature': x, 'label': y})
        .make_one_shot_iterator().get_next())
lda_bert = LatentDirichletAllocation(n_components = 3, learning_offset = 50., verbose=2)
t0 = time()
lda_bert.fit(bert_vectors)
print("BERT: done in %0.3fs" % (time() - t0))
with open('uncased_L-12_H-768_A-12/vocab.txt') as vocab_file:
    bert_feature_names = vocab_file.read().splitlines()
print('Topics from LDA using BERT:')
print_top_words(lda_bert, bert_feature_names, 25)

