"""Lyrics classifier for Flask application"""

# Our familiar imports  
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import scipy.sparse as sp


# A new import
import pickle

class LyricsClf():
    """A MultinomialNB classifier for predicting artists from lyrics.
    Offers train, save, and load routines for offline and startup
    purposes. Offers predictArtist for online use.
    """
    def __init__(self,picklefile=None):
        """Constructor that creates an empty artistLabels dictionary,
        a CountVectorizer placeholder, and a classifier placeholder.
        If a picklefile is specified, the returned object is instantiated
        from a pickled version on disk.
        """
        self.artistLabels = dict()
        self.vectorizer = None
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

    def predictArtist(self,lyrics):
        """Returns an artist name given sample song lyrics.
        Applies the Million Song Dataset stemming routine to
        the lyrics (pre-processing), vectorizes the lyrics,
        and runs them through the MultinomialNB classifier.
        Returns the artist name associated with the predicted
        label.
        """

        df = pd.DataFrame({'Lyrics':[lyrics]})
        X = self.unigram_feature_names.transform(df['Lyrics'])
        Y = self.bigram_feature_names.transform(df['Lyrics'])
        Z = self.trigram_feature_names.transform(df['Lyrics'])
        business_trigrams_and_bigrams = sp.hstack((Z, Y), format='csr')
        business_all_grams = sp.hstack((business_trigrams_and_bigrams, X), format='csr')
        predictedRating = self.clf.predict(business_all_grams)
        return predictedRating

    def load(self,picklefile):
        """Load a LyricsClf object from picklefile.
        Return this loaded object for future use.
        """
        loaded = pickle.load(open(picklefile,'rb'))
        self.artistLabels = loaded.artistLabels
        self.vectorizer = loaded.vectorizer
        self.clf = loaded.clf
        return self

# end of LyricsClf class
