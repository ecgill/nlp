import nltk
import random
import numpy as np
from bs4 import BeautifulSoup

def load_reviews(posfn, negfn, key, balance=False):
    positive_reviews = BeautifulSoup(open(posfn).read(), 'lxml').findAll(key)
    negative_reviews = BeautifulSoup(open(negfn).read(), 'lxml').findAll(key)
    if balance:
        np.random.shuffle(positive_reviews)
        positive_reviews = positive_reviews[:len(negative_reviews)]
    return positive_reviews, negative_reviews

def find_trigrams(reviews):
    trigrams = {}
    for review in reviews:
        s = review.text.lower()
        tokens = nltk.tokenize.word_tokenize(s)
        for i in range(len(tokens) - 2):
            k = (tokens[i], tokens[i+2])
            if k not in trigrams:
                trigrams[k] = []
            trigrams[k].append(tokens[i+1])
    return trigrams

def calc_trigram_probs(trigrams):
    trigram_probs = {}
    for k, words in trigrams.items():
        if len(set(words)) > 1:
            d = {}
            n = 0
            for w in words:
                if w not in d:
                    d[w] = 0
                d[w] += 1
                n += 1
            for w, c in d.items():
                d[w] = float(c)/n
            trigram_probs[k] = d
    return trigram_probs

def random_sample(d):
    r = random.random()
    cumulative = 0
    for w, p in d.items():
        cumulative += p
        if r < cumulative:
            return w

def test_spinner(pos, trigram_probs):
    review = random.choice(pos)
    s = review.text.lower()
    print('Original: ', s)
    tokens = nltk.tokenize.word_tokenize(s)
    for i in range(len(tokens) - 2):
        if random.random() < 0.2:
            k = (tokens[i], tokens[i+2])
            if k in trigram_probs:
                w = random_sample(trigram_probs[k])
                tokens[i+1] = w
    print('Spun: ', )
    print(' '.join(tokens).replace(' .', '.').replace(" '", "'").replace(' ,', ',').replace(' !', '!'))

def main():
    # Read in positive and negative reviews, strip XML
    pos, neg  = load_reviews(posfn='data/positive.review.txt',
                             negfn='data/negative.review.txt',
                             key='review_text',
                             balance=False)
    trigrams = find_trigrams(pos)
    trigram_probs = calc_trigram_probs(trigrams)
    test_spinner(pos, trigram_probs)

if __name__ == '__main__':
    main()
