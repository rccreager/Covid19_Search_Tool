from pathlib import Path, PurePath
import pandas as pd
import nltk
import re 
from time import time

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 

from src import nlp

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "\nTopic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

input_dir = PurePath('data/CORD-19-research-challenge/')
list(Path(input_dir).glob('*'))

metadata_path = input_dir / 'metadata.csv'
metadata = pd.read_csv(metadata_path,
                               dtype={'Microsoft Academic Paper ID': str,
                                      'pubmed_id': str})

# Set the abstract to the paper title if it is null
metadata.abstract = metadata.abstract.fillna(metadata.title)

duplicate_paper = ~(metadata.title.isnull() | metadata.abstract.isnull()) & (metadata.duplicated(subset=['title', 'abstract']))
metadata = metadata[~duplicate_paper].reset_index(drop=True)
drop_columns = ['authors',
                'sha',
                'has_full_text',
                'full_text_file',
                'Microsoft Academic Paper ID',
                'WHO #Covidence', 
                'pmcid', 
                'pubmed_id', 
                'license']
metadata = metadata.drop(axis=1,labels=drop_columns)
metadata = metadata.dropna()
english_stopwords = list(set(stopwords.words('english')))
preprocessed = [nlp.preprocess(text) for text in metadata['abstract']]

tf_vectorizer = CountVectorizer(min_df=3, max_df=0.1, stop_words=english_stopwords)
tf_vectors = tf_vectorizer.fit_transform(preprocessed)

tfidf_vectorizer = TfidfVectorizer()
tfidf_vectors = tfidf_vectorizer.fit_transform(preprocessed)

lda_tf = LatentDirichletAllocation(n_components = 3, learning_offset = 50., verbose=2)
lda_tfidf = LatentDirichletAllocation(n_components = 3, learning_offset = 50., verbose=2)

t0 = time()
lda_tf.fit(tf_vectors)
lda_tfidf.fit(tfidf_vectors)
print("done in %0.3fs" % (time() - t0))

tf_feature_names = tf_vectorizer.get_feature_names()
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print('Topics from LDA using term frequency (TF):')
print_top_words(lda_tf, tf_feature_names, 25)
print('Topics from LDA using TF-IDF:')
print_top_words(lda_tfidf, tfidf_feature_names, 25)
