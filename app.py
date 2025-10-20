# app.py
from flask import Flask, render_template, request
import pickle
import string
import nltk
from nltk.corpus import stopwords

# Initialize Flask app
app = Flask(__name__)

# Load model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load stopwords once
nltk.download('stopwords', quiet=True)
stop_words = stopwords.words('english')

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    headline = request.form['headline']
    processed = preprocess(headline)
    vectorized = vectorizer.transform([processed])
    prediction = model.predict(vectorized)[0]
    result = "Real" if prediction == 1 else "Fake"
    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
