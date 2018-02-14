# Emily Gill February 2018
# Data from https://www.cs.jhu.edu/~mdredze/datasets/sentiment/

import nltk
import numpy as np
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression

def load_stopwords(filename):
    return set(w.rstrip() for w in open(filename))

def load_reviews(posfn, negfn, key, balance=False):
    positive_reviews = BeautifulSoup(open(posfn).read(), 'lxml').findAll(key)
    negative_reviews = BeautifulSoup(open(negfn).read(), 'lxml').findAll(key)
    if balance:
        np.random.shuffle(positive_reviews)
        positive_reviews = positive_reviews[:len(negative_reviews)]
    return positive_reviews, negative_reviews

def tokenizer(s, stopwords):
    s = s.lower()                                           # lowercase all
    tokens = nltk.tokenize.word_tokenize(s)                 # use nltk tokenizer
    tokens = [t for t in tokens if len(t) > 2]              # remove short words
    wordnet_lemm = WordNetLemmatizer()                      # lemmatizer
    tokens = [wordnet_lemm.lemmatize(t) for t in tokens]   # turns words to baseform
    tokens = [t for t in tokens if t not in stopwords]      # remove stopwords
    return tokens

def create_word_index(pos_reviews, neg_reviews, stopwords):
    word_index = {}
    current_index = 0
    pos_collect_tokens = []
    for review in pos_reviews:
        tokens = tokenizer(review.text, stopwords)
        pos_collect_tokens.append(tokens)
        for token in tokens:
            if token not in word_index:
                word_index[token] = current_index
                current_index += 1
    neg_collect_tokens = []
    for review in neg_reviews:
        tokens = tokenizer(review.text, stopwords)
        neg_collect_tokens.append(tokens)
        for token in tokens:
            if token not in word_index:
                word_index[token] = current_index
                current_index += 1

    return word_index, pos_collect_tokens, neg_collect_tokens

def tokens_to_vector(tokens, word_index, label):
    vec = np.zeros(len(word_index) + 1)
    for t in tokens:
        i = word_index[t]
        vec[i] += 1
    vec = vec/vec.sum()
    vec[-1] = label
    return vec

def tokens_to_matrix(word_index, pos_tokens, neg_tokens):
    N = len(pos_tokens) + len(neg_tokens)
    data = np.zeros((N, len(word_index) + 1))
    i = 0
    for tokens in pos_tokens:
        vec = tokens_to_vector(tokens, word_index, 1)
        data[i,:] = vec
        i += 1
    for tokens in neg_tokens:
        vec = tokens_to_vector(tokens, word_index, 0)
        data[i,:] = vec
        i += 1
    return data

def split_data(data):
    np.random.shuffle(data)
    X = data[:, :-1]
    y = data[:, -1]
    X_train = X[:-200,]
    y_train = y[:-200,]
    X_test = X[-200:,]
    y_test = y[-200:,]
    return X_train, X_test, y_train, y_test

def plot_wordclouds(word_freqs, label):
    wc_plot = WordCloud(width=512, height=512).generate_from_frequencies(word_freqs)
    plt.figure()
    plt.imshow(wc_plot)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig('img/sentiment_wordclouds_' + label + '.png')

def main():
    # Read in pre-defined list of stopwords
    stopwords = load_stopwords(filename='data/stopwords.txt')

    # Read in positive and negative reviews, strip XML
    pos, neg  = load_reviews(posfn='data/positive.review.txt',
                             negfn='data/negative.review.txt',
                             key='review_text',
                             balance=True)

    # Tokenize reviews and create dictionary of words and their index
    word_index, pos_tokens, neg_tokens = create_word_index(pos,
                                                           neg,
                                                           stopwords)

    # Create word-vector matrix ready for classification
    data = tokens_to_matrix(word_index, pos_tokens, neg_tokens)

    # Split data for training and validation sets
    X_train, X_test, y_train, y_test = split_data(data)

    # Fit classification model and print metric accuracy
    model = LogisticRegression()
    model.fit(X_train, y_train)
    print('Training set accuracy: {:.3f}'.format(model.score(X_train, y_train)))
    print('Testing set accuracy:  {:.3f}'.format(model.score(X_test, y_test)))

    # Define a threshold for "positive" and "negative" sentiment
    threshold = 0.5
    pos_words = {}
    neg_words = {}
    for word, index in word_index.items():
        weight = model.coef_[0][index]
        if weight > threshold:
            pos_words[word] = weight
        if weight < -threshold:
            neg_words[word] = weight

    # Plot and save wordclouds for pos/neg sentiment words
    plot_wordclouds(pos_words, label='pos')
    plot_wordclouds(neg_words, label='neg')

if __name__ == '__main__':
    main()
