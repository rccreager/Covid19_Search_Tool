import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path, PurePath
import pandas as pd
import requests
from requests.exceptions import HTTPError, ConnectionError
from rank_bm25 import BM25Okapi
import nltk
nltk.download("punkt")
import re 
from time import time

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "\nTopic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

input_dir = PurePath('./CORD-19-research-challenge')
list(Path(input_dir).glob('*'))

metadata_path = input_dir / '../metadata.csv'
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
print(metadata.shape)
print(metadata.head(10))

#count_vect = CountVectorizer(stop_words='english')
#X_train_counts = count_vect.fit_transform(metadata['abstract'] )
#tfidf_transformer = TfidfTransformer()
#X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
nltk.download("stopwords")
def strip_characters(text):
    t = re.sub('\(|\)|:|,|;|\.|’|”|“|\?|%|>|<', '', text)
    t = re.sub('/', ' ', t)
    t = t.replace("'",'')
    return t

def clean(text):
    t = text.lower()
    t = strip_characters(t)
    return t

def tokenize(text):
    words = nltk.word_tokenize(text)
    return list(set([word for word in words 
                     if len(word) > 1
                     and not word in english_stopwords
                     and not (word.isnumeric() and len(word) is not 4)
                     and (not word.isnumeric() or word.isalpha())] )
               )
    
def lemmatize(word_list,lemmatizer):
    # Init the Wordnet Lemmatizer
    lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
    return lemmatized_output


def preprocess(text):
    t = clean(text)
    tokens = tokenize(t)
    lemmatizer=WordNetLemmatizer()
    tokens = lemmatize(tokens,lemmatizer)
    return tokens

english_stopwords = list(set(stopwords.words('english')))
medical_stopwords = ['sars', 'cov', 'wuhan', 'china', 'ncov', 'january', 'december', 'health', 'case', 'patient', 'public', 
                     'human', 'disease', 'first', 'outbreak', 'transmission', 'reported', 'respiratory', 'pneumonia', 'coronavirus', '2019']
for med_word in medical_stopwords:
  english_stopwords.append(med_word)

preprocessed = [preprocess(text) for text in metadata['abstract']]
print(preprocessed[0])
lda_tf = CountVectorizer(min_df=2, stop_words=english_stopwords)

word_counts = lda_tf.fit_transform([preprocess(text) for text in metadata['abstract']])

lda_tfidf = TfidfTransformer()
lda_tfidf.fit(word_counts)

lda = LatentDirichletAllocation(n_components = 5, learning_offset = 50., verbose=2)

lda_pipeline = Pipeline([
    ('vect', lda_tf),
    ('tfidf', lda_tfidf),
    ('clf', lda),
])
t0 = time()
lda_pipeline.fit([preprocess(text) for text in metadata['abstract']])
print("done in %0.3fs" % (time() - t0))
lda = LatentDirichletAllocation(learning_offset = 50., verbose=2)
lda_pipeline = Pipeline([
    ('vect', lda_tf),
    ('tfidf', lda_tfidf),
    ('clf', lda),
])
parameters_lda = {
    'vect__max_df': (0.1, 0.25, 0.5, 0.99),
    'vect__min_df': (2, 3),
    'clf__n_components': (3, 5, 10),
}
grid_search_lda = GridSearchCV(lda_pipeline, parameters_lda, cv=2, n_jobs=2, verbose=2)
print("Performing LDA grid search...")
print("parameters_lda:")
print(parameters_lda)
t0 = time()
grid_search_lda.fit(metadata['abstract'])
print("done in %0.3fs" % (time() - t0))
print(grid_search_lda.cv_results_)
print("Best score: %0.3f" % grid_search_lda.best_score_)
print("Best parameters_lda set:")
best_parameters_lda = grid_search_lda.best_estimator_.get_params()
best_clf_lda = grid_search_lda.best_estimator_
for param_name in sorted(parameters_lda.keys()):
    print("\t%s: %r" % (param_name, best_parameters_lda[param_name]))

tf_feature_names = lda_tf.get_feature_names()
print_top_words(best_clf_lda, tf_feature_names, 25)
