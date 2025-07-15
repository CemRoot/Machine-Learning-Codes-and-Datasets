# nlp_pipeline_advanced.py

import re, pandas as pd, nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

nltk.download('stopwords', quiet=True)
stemmer = PorterStemmer()
stops = set(stopwords.words('english')) - {'not'}

def preprocess(texts):
    cleaned = []
    for doc in texts:
        letters = re.sub('[^a-zA-Z]', ' ', doc)
        tokens = letters.lower().split()
        filtered = [stemmer.stem(w) for w in tokens if w not in stops]
        cleaned.append(' '.join(filtered))
    return cleaned

# 1. Veri yükleme & ön işleme
df = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
corpus = preprocess(df['Review'])
y = df['Liked'].values

# 2. Eğitim/Test ayrımı
X_train, X_test, y_train, y_test = train_test_split(
    corpus, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Pipeline + GridSearchCV tanımı
pipe = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('var', VarianceThreshold(0.0)),
    ('clf', MultinomialNB())
])

param_grid = [
    {   # MultinomialNB
        'tfidf__max_features': [1000,1500,2000],
        'tfidf__ngram_range': [(1,1),(1,2)],
        'tfidf__min_df': [3,5,10],
        'tfidf__max_df': [0.7,0.8,0.9],
        'clf': [MultinomialNB()],
        'clf__alpha': [0.1,0.5,1.0]
    },
    {   # LinearSVC
        'tfidf__max_features': [1500,2000],
        'tfidf__ngram_range': [(1,1),(1,2)],
        'clf': [LinearSVC(max_iter=10000)],
        'clf__C': [0.1,1,10]
    },
    {   # LogisticRegression
        'tfidf__max_features': [1500,2000],
        'tfidf__ngram_range': [(1,1),(1,2)],
        'clf': [LogisticRegression(solver='liblinear', max_iter=10000)],
        'clf__C': [0.1,1,10]
    }
]

grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

# 4. En iyi model ile değerlendirme
best = grid.best_estimator_
y_pred = best.predict(X_test)
print("Best params:", grid.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))