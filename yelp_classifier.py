# Our familiar imports  
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import scipy.sparse as sp


# A new import
import pickle

class YelpClf():
    def __init__(self,picklefile=None):
        self.clf = None
        f = open('business_unigram_feature_names', 'r')
        self.unigram_feature_names = pickle.load(f)
        f = open('bigram_feature_names', 'r')
        self.bigram_feature_names = pickle.load(f)
        f = open('trigam_vocab', 'r')
        trigram_vocab = pickle.load(f)
        self.trigram_feature_names = CountVectorizer(vocabulary=trigram_vocab, ngram_range=(3,3))
        f = open('logreg_business_rating_classifier', 'r')
        self.clf = pickle.load(f)

    def predictRating(self,reviews):
        df = pd.DataFrame({'Reviews':[reviews]})
        X = self.unigram_feature_names.transform(df['Reviews'])
        Y = self.bigram_feature_names.transform(df['Reviews'])
        Z = self.trigram_feature_names.transform(df['Reviews'])
        business_trigrams_and_bigrams = sp.hstack((Z, Y), format='csr')
        business_all_grams = sp.hstack((business_trigrams_and_bigrams, X), format='csr')
        predictedRating = self.clf.predict(business_all_grams)
        return predictedRating

    def load(self,picklefile):
        loaded = pickle.load(open(picklefile,'rb'))
        self.clf = loaded.clf
        return self
