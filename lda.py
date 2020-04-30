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
from bert import get_bert_vectorizer
from bert_utils import get_feed_dict

preprocessed = get_preprocessed_abstract_text('data/CORD-19-research-challenge/', 'metadata.csv')


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

preprocessed = get_preprocessed_abstract_text('data/CORD-19-research-challenge/', 'metadata.csv')
print(preprocessed[0])
print('getting client')
with BertClient(ip = '1.2.4.8') as bc:
    print('encoding')
    bc.encode(preprocessed, is_tokenized=False)
print('fetching')
bert_vectors = bc.fetch_all(sort=True)

lda_bert = LatentDirichletAllocation(n_components = 3, learning_offset = 50., verbose=2)
t0 = time()
lda_bert.fit(bert_vectors)
print("BERT: done in %0.3fs" % (time() - t0))
with open('uncased_L-12_H-768_A-12/vocab.txt') as vocab_file:
    bert_feature_names = vocab_file.read().splitlines()
print('Topics from LDA using BERT:')
print_top_words(lda_bert, bert_feature_names, 25)

