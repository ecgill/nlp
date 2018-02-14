import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD

def load_titles(filename):
    return [line.rstrip() for line in open(filename)]

def load_stopwords(filename):
    stopwords = set(w.rstrip() for w in open(filename))
    stopwords = stopwords.union({
        'introduction', 'edition', 'series', 'application',
        'approach', 'card', 'access', 'package', 'plus', 'etext',
        'brief', 'vol', 'fundamental', 'guide', 'essential', 'printed',
        'third', 'second', 'fourth', })
    return stopwords

def tokenizer(s, stopwords):
    print(s)
    s = s.lower()                                           # lowercase all
    tokens = nltk.tokenize.word_tokenize(s)                 # use nltk tokenizer
    tokens = [t for t in tokens if len(t) > 2]              # remove short words
    wordnet_lemm = WordNetLemmatizer()                      # lemmatizer
    tokens = [wordnet_lemm.lemmatize(t) for t in tokens]   # turns words to baseform
    tokens = [t for t in tokens if t not in stopwords]      # remove stopwords
    tokens = [t for t in tokens if not any(c.isdigit() for c in t)] # remove #s
    return tokens

def create_word_index(titles, stopwords):
    word_index_map = {}
    current_index = 0
    all_tokens = []
    all_titles = []
    index_word_map = []
    error_count = 0
    for title in titles:
        try:
            title = title.encode('ascii', 'ignore').decode('utf-8') # this will throw exception if bad characters
            all_titles.append(title)
            tokens = tokenizer(title, stopwords)
            all_tokens.append(tokens)
            for token in tokens:
                if token not in word_index_map:
                    word_index_map[token] = current_index
                    current_index += 1
                    index_word_map.append(token)
        except Exception as e:
            print(e)
            print(title)
            error_count += 1
    return word_index_map, all_tokens, index_word_map

def tokens_to_vector(tokens, word_index_map):
    vec = np.zeros(len(word_index_map))
    for t in tokens:
        i = word_index_map[t]
        vec[i] += 1
    return vec

def tokens_to_matrix(word_index_map, all_tokens):
    N = len(all_tokens)
    D = len(word_index_map)
    data = np.zeros((D, N)) # terms=rows, docs = cols
    i = 0
    for tokens in all_tokens:
        vec = tokens_to_vector(tokens, word_index_map)
        data[:,i] = vec
        i += 1
    return data

def main():
    title_fn = 'data/all_book_titles.txt'
    titles = load_titles(title_fn)

    stopwords_fn = 'data/stopwords.txt'
    stopwords = load_stopwords(stopwords_fn)

    word_index_map, all_tokens, index_word_map = create_word_index(titles, stopwords)
    data = tokens_to_matrix(word_index_map, all_tokens)

    D = len(word_index_map)
    svd = TruncatedSVD()
    Z = svd.fit_transform(data)
    plt.scatter(Z[:,0], Z[:,1])
    for i in range(D):
        plt.annotate(s=index_word_map[i], xy=(Z[i,0], Z[i,1]))
    plt.savefig('img/SVD.png')
    # plot shows history and science far apart.
    # computer, biology and science close together
    # america, global, book, world, modern, history all close together

if __name__ == '__main__':
    main()
