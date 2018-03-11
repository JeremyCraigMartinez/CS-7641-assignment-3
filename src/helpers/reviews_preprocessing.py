from os.path import dirname, realpath
import re

from sklearn.feature_extraction.text import CountVectorizer
import sklearn.model_selection as ms
import pandas as pd

# uncomment to download stopwords
#import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

dir_path = dirname(realpath(__file__))

def get_corpus(dataset):
    corpus = []
    for i in range(0, 1000):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)
    return corpus

# Creating the Bag of Words model
def get_X_Y(corpus, dataset):
    cv = CountVectorizer(max_features=1500)
    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, 1].values

    return X, y

# Splitting the dataset into the Training set and Test set
def get_train_test_set():
    # Importing the dataset
    dataset = pd.read_csv('{}/../../data/Restaurant_Reviews.tsv'.format(dir_path), delimiter='\t', quoting=3)

    corpus = get_corpus(dataset)
    X, y = get_X_Y(corpus, dataset)

    # X_train, X_test, y_train, y_test
    return ms.train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)