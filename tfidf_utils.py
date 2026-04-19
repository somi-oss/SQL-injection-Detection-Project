from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf_features(X_train, X_test):
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        max_features=20000
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf
