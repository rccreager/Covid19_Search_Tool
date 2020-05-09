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

preprocessed = get_preprocessed_abstract_text('data/CORD-19-research-challenge/', 'metadata.csv', tokenize_text = True)
english_stopwords = list(set(stopwords.words('english')))

tf_vectorizer = CountVectorizer(min_df=3, max_df=0.1, stop_words=english_stopwords)
tf_vectors = tf_vectorizer.fit_transform(preprocessed)
lda_tf = LatentDirichletAllocation(n_components = 3, learning_offset = 50., verbose=2)
t0 = time()
lda_tf.fit(tf_vectors)
print("TF: done in %0.3fs" % (time() - t0))
tf_feature_names = tf_vectorizer.get_feature_names()
print('Topics from LDA using term frequency (TF):')
print_top_words(lda_tf, tf_feature_names, 25)

tfidf_vectorizer = TfidfVectorizer(stop_words=english_stopwords)
tfidf_vectors = tfidf_vectorizer.fit_transform(preprocessed)
lda_tfidf = LatentDirichletAllocation(n_components = 3, learning_offset = 50., verbose=2)
t0 = time()
lda_tfidf.fit(tfidf_vectors)
print("TFIDF: done in %0.3fs" % (time() - t0))
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print('Topics from LDA using TF-IDF:')
print_top_words(lda_tfidf, tfidf_feature_names, 25)

#
