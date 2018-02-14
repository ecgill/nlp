import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import precision_score

def models_to_compare():
    models = []
    models.append(['MNB', MultinomialNB()])
    models.append(['GNB', GaussianNB()])
    models.append(['BNB', BernoulliNB()])
    models.append(['AB', AdaBoostClassifier()])
    models.append(['RF', RandomForestClassifier(n_estimators=250)])
    models.append(['QDA', QuadraticDiscriminantAnalysis()])
    models.append(['KNN', KNeighborsClassifier(10)])
    models.append(['SVM', SVC(kernel="linear", C=0.025)])
    models.append(['MLP', MLPClassifier(alpha=1)])
    return models

def compare_models(X, y, models):
    seed = 123
    results = []
    names = []
    scoring = 'precision' # we want to minimize FP = optimize precision
    for name, model in models:
    	kfold = KFold(n_splits=10, random_state=seed)
    	cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    	results.append(cv_results)
    	names.append(name)
    	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    	print(msg)
    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison ({})'.format(scoring))
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.savefig('img/spam_crossval_' + scoring + '.png')

def load_data(filename):
    data = []
    f = open(filename)
    reader = csv.reader(f)
    # next(reader, None)
    for row in reader:
        data.append(row)
    f.close()

    X = np.array([x[:-1] for x in data]).astype(np.float)
    y = np.array([x[-1] for x in data]).astype(np.float)
    del data
    return X[:,:48], y

def main():
    print('--- Reading in data...')
    filename = 'data/spambase.data.txt'
    X, y = load_data(filename)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)
    print('--- Comparing a bunch of classifiers out of the box and plot...')
    # models = models_to_compare()
    # compare_models(X_train, y_train, models)
    print('--- Best model from gridsearch RF...')
    rf_mod = RandomForestClassifier(n_estimators=325,
                                    max_features=1,
                                    min_samples_leaf=1,
                                    max_depth=None,
                                    criterion='gini')

    print('--- Fitting on training data and validate on test data...')
    rf_mod.fit(X_train, y_train)
    y_pred_train = rf_mod.predict(X_train)
    y_pred_test = rf_mod.predict(X_test)
    print('Training set precision score:   {:.3f}'.format(precision_score(y_train, y_pred_train)))
    print('Validation set precision score: {:.3f}'.format(precision_score(y_test, y_pred_test)))

    print('--- Metrics of full dataset...')
    rf_mod = RandomForestClassifier(n_estimators=325,
                                    max_features=1,
                                    min_samples_leaf=1,
                                    max_depth=None,
                                    criterion='gini')
    rf_mod.fit(X, y)
    y_pred = rf_mod.predict(X)
    print('Full data score:                {:.3f}'.format(precision_score(y, y_pred)))




if __name__ == "__main__":
    main()

    # word_labels = ['address', 'all', '3d', 'our', 'over', 'remove', 'internet',
    #            'order', 'mail', 'receive', 'will', 'people', 'report',
    #            'addresses','free', 'business', 'email', 'you', 'credit', 'your',
    #            'font', '000', 'money', 'hp', 'hpl', 'george', '650', 'lab',
    #            'labs', 'telnet', '857', 'data', '415', '85', 'technology',
    #            '1999', 'parts', 'pm', 'direct', 'cs', 'meeting', 'original',
    #            'project', 're', 'edu', 'table', 'conference' , 'y']



    # mlp_mod = MLPClassifier()
    # mlp_params = {'activation': ['relu', 'logistic', 'identity'],
    #                 'alpha': [0.0001, 0.001, 0.01],
    #                 'learning_rate': ['constant', 'adaptive'],
    #                 'hidden_layer_sizes': [(100,), (100,50), (100,50,50)]}
    # mlp_grid = GridSearchCV(mlp_mod, param_grid=mlp_params)
    # mlp_grid.fit(X_train, y_train)
    # {'activation': 'relu',
    # 'alpha': 0.001,
    # 'hidden_layer_sizes': (100, 50, 50),
    # 'learning_rate': 'constant'}
