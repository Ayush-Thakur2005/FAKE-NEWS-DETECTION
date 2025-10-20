# model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import nltk
from nltk.corpus import stopwords
import string
import os

# Download stopwords if not already
nltk.download('stopwords', quiet=True)
stop_words = stopwords.words('english')

# Paths
csv_path = 'dataset/FakeNewsNet.csv'
preprocessed_path = 'dataset/preprocessed.csv'

# 1️⃣ Load dataset
if os.path.exists(preprocessed_path):
    df = pd.read_csv(preprocessed_path)
else:
    df = pd.read_csv(csv_path)
    df = df.fillna('')

    # Preprocessing function
    def preprocess(text):
        text = text.lower()
        text = ''.join([c for c in text if c not in string.punctuation])
        words = text.split()
        words = [word for word in words if word not in stop_words]
        return ' '.join(words)

    df['content'] = df['title'].apply(preprocess)
    df.to_csv(preprocessed_path, index=False)

# 2️⃣ Features and labels
X = df['content']
y = df['real']

# 3️⃣ Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4️⃣ TF-IDF vectorization (smaller feature set)
vectorizer = TfidfVectorizer(max_features=2000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5️⃣ Logistic Regression with faster solver
model = LogisticRegression(solver='saga', max_iter=1000)
model.fit(X_train_tfidf, y_train)

# 6️⃣ Evaluate
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# 7️⃣ Save model and vectorizer
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
