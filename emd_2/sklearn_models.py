from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.experimental import enable_halving_search_cv # noqa
from sklearn.model_selection import HalvingGridSearchCV
import pickle

from data import get_all_data, score_metric, normalize_review_weight

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = get_all_data()
    train_weights = [normalize_review_weight(w) for w in X_train['helpful']]

    tfidf_grid = {
        'vectorizer__lowercase': [True, False],
        'vectorizer__ngram_range': [(1, 3), (1, 4), (2, 4)],
        'vectorizer__max_df': [1.0, 0.95, 0.9, 0.85, 0.8],
        'vectorizer__min_df': [25, 50, 100, 200, 0.01, 0.05],
    }

    svm = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', LinearSVC(
            class_weight='balanced'))
    ])
    grid_search = HalvingGridSearchCV(svm, tfidf_grid, random_state=42, verbose=10, n_jobs=12)
    grid_search.fit(X_train['reviewText'], y_train, classifier__sample_weight=train_weights)

    print(grid_search.best_params_)
    print(score_metric(y_test, grid_search.best_estimator_.predict(X_test['reviewText'])))
    with open('model/sklearn-svc.pkl', 'wb') as out:
        pickle.dump(grid_search.best_estimator_, out)


    bayes = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', MultinomialNB(
            fit_prior=True))
    ])
    grid_search = HalvingGridSearchCV(bayes, tfidf_grid, random_state=42, verbose=10, n_jobs=4)
    grid_search.fit(X_train['reviewText'], y_train, classifier__sample_weight=train_weights)

    print(grid_search.best_params_)
    print(score_metric(y_test, grid_search.best_estimator_.predict(X_test['reviewText'])))
    with open('model/sklearn-bayes.pkl', 'wb') as out:
        pickle.dump(grid_search.best_estimator_, out)