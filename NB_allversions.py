'''
Testing the following versions of Naive Bayes
1. Naïve Bayes with present features
2. Naïve Bayes with present and absent features
3. Naïve Bayes with present and absent features with Laplace smoothing
4. Expectation Maximization of Naïve Bayes with present and absent features with Laplace smoothing
5. Complement of Naïve Bayes with present and absent features with Laplace smoothing
6. Expectation Maximization of complement of Naïve Bayes with present and absent features with Laplace smoothing
'''

import naive_bayes
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re
import codecs
import numpy as np
import itertools
import seminb
import semiCnb
import pandas as pd


class docparser(object):
    """
    parser to get the CSV dataset
    """

    def __init__(self):
        pass

    def get_data(self, file_path, Text=True):
        if Text:
            data = []
            labels = []
            f = codecs.open(file_path, 'r', encoding="utf8", errors='ignore')
            for line in f:
                doc, label = self.parse_line(line)
                data.append(doc)
                labels.append(label)
            return data, np.array(labels)
        else:
            df = pd.read_csv(file_path)
            df.columns = ['Label', 'Rating', 'Review']
            rating = pd.get_dummies(df.loc[:, 'Rating'])
            text = []
            labels = []
            for ii, ij in zip(df.loc[:, 'Review'].values, df.loc[:, 'Label'].values):
                # print(ii, ij)
                if ii == ' ':
                    pass
                else:
                    text.append(ii)
                    labels.append(ij)
            return text, np.array(labels), rating

if __name__ == '__main__':
    # ---------Global parameters-----------------
    Text = False
    max_features = None
    n_sets = 10
    set_size = 1.0 / n_sets
    cumulative_percent = 0
    # set project directory
    abs_path = 'C:/Users/malsa876/Desktop/Pytest/mytracks_NaiveBayes_Filter.csv'     #mytracks_NaiveBayes_Filter #flutter-reviews-NB.csv
    if Text:
        pass
    else:
        data, labels, rating = docparser().get_data(abs_path, Text)

        # ----Naive Bayes with present features only
        labeled_reviews = naive_bayes.get_labeled_reviews(abs_path)
        all_words = {}
        for (r, label) in labeled_reviews:
            for word in r.split(" "):
                if len(word) > 1:
                    all_words[word] = 0
        # featureset for NB
        featuresets = [(naive_bayes.review_features(r, {}), label) for (r, label) in labeled_reviews]
        featureset = [(naive_bayes.review_features(r, all_words), label) for (r, label) in labeled_reviews]
        print(featuresets[23])
        print(featureset[23])
        #print(featuresets)
        #print(featureset)

        '''
        print('Start Naive Bayes Classification with present features only')
        naive_bayes.cross_validation(featuresets, n_sets, NB=True)
        print('End of Naive Bayes Classification with present features only\n')
        print('*' * 40)


    # ----Naive Bayes with present and absent features without Laplace Smoothing
        labeled_reviews = naive_bayes.get_labeled_reviews(abs_path)
        all_words = {}
        for (r, label) in labeled_reviews:
            for word in r.split(" "):
                if len(word) > 1:
                    all_words[word] = 0
        # featureset for NB
        featuresets = [(naive_bayes.review_features(r, all_words), label) for (r, label) in labeled_reviews]
        print('Start Naive Bayes Classification with present and absent features without Laplace Smoothing')
        naive_bayes.cross_validation(featuresets, n_sets, NB=True)
        print('End of Naive Bayes Classification with present and absent features without Laplace Smoothing\n')
        print('*' * 40)

        '''
        # ---- Naive Bayes with present and absent features with Laplace Smoothing
        labeled_reviews = naive_bayes.get_labeled_reviews(abs_path)
        words = {}
        for (r, label) in labeled_reviews:
            for word in r.split(" "):
                if len(word) > 1:
                    all_words[word] = 0
        # featureset for NB
        featureset = [(naive_bayes.review_features(r, {}), label) for (r, label) in labeled_reviews]
        print('Start of Naive Bayes Classification with present and absent features with laplace smoothing')
        naive_bayes.cross_validation(featureset, n_sets, mNB=True, alpha = 1.0)
        print('End of Naive Bayes Classification with present and absent features with laplace smoothing\n')
        print('*' * 40)

        # ---- Complement Naive Bayes with present and absent features with laplace smoothing
        '''
        labeled_reviews = naive_bayes.get_labeled_reviews(abs_path)
        words = {}
        for (r, label) in labeled_reviews:
            for word in r.split(" "):
                if len(word) > 1:
                    all_words[word] = 0
        # featureset for NB
        featureset = [(naive_bayes.review_features(r, all_words), label) for (r, label) in labeled_reviews]
        print('Start of Complement Naive Bayes with present and absent features with laplace smoothing')
        naive_bayes.cross_validation(featureset, n_sets, cNB=True, alpha=1.0)
        print('End of Complement Naive Bayes with present and absent features with laplace smoothing\n')
        print('*' * 40)

        # Expectation Maximization version of Naive Bayes Classification with present and absent features and laplace smoothing
        print(
            'Begin Expectation Maximization version of Naive Bayes Classification with present and absent features with laplace smoothing')
        naive_bayes.cross_validation(featureset, n_sets, emNB=True, alpha=1.0)
        print(
            'End of Expectation Maximization version of Naive Bayes Classification with present and absent features with laplace smoothing\n')
        print('*' * 40)

        # Expectation Maximization version of Complement Naive Bayes Classification with present and absent features and laplace smoothing
        print(
            'Begin Expectation Maximization version of Complement Naive Bayes Classification with present and absent features with laplace smoothing')
        naive_bayes.cross_validation(featureset, n_sets, emCNM=True, alpha=1.0)
        print(
            'End of Expectation Maximization version of Complement Naive Bayes Classification with present and absent features with laplace smoothing\n')
        print('*' * 40)
        '''
        '''
        #VII EM of NB with present features
        naive_bayes.cross_validation(featuresets, n_sets, emNB=True)
        '''

        '''
        #VIII NB with present features and laplace smoothing
        naive_bayes.cross_validation(featuresets, n_sets, mNB=True, alpha=1.0)
        '''

        '''
        #IX EM of NB with present features and laplace smoothing
        naive_bayes.cross_validation(featuresets, n_sets, emNB=True, alpha=1.0)
        '''

        '''
        #X EM with present and absent features
        naive_bayes.cross_validation(featureset, n_sets, emNB=True)
        '''
        '''
        #XI Complement NB with present features
        naive_bayes.cross_validation(featuresets, n_sets, cNB=True)
        '''

        '''
        #XII EM of complement NB with present features
        naive_bayes.cross_validation(featuresets, n_sets, emCNM=True)
        '''

        '''
        #XIII Complement NB with present features and Laplace Smoothing
        naive_bayes.cross_validation(featuresets, n_sets, cNB=True, alpha=1.0)
        '''
        '''
        #XIV EM of Complement NB with present features and Laplace Smoothing
        naive_bayes.cross_validation(featuresets, n_sets, emCNM=True, alpha=1.0)
        '''

        '''
        #XV Complement NB with present and absent features
        naive_bayes.cross_validation(featureset, n_sets, cNB=True)
        '''

        '''
        #XVI EM of Compelement NB with present and absent features
        naive_bayes.cross_validation(featureset, n_sets, emCNM=True)
        '''