from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
from data_preprocessing import preprocess_data

app = Flask(__name__)

# Global variables for the model and vectorizer
model = None
vectorizer = None

def train_model():
    """Train the email classification model"""
    global model, vectorizer
    
    # Load and preprocess the data
    print("Loading and preprocessing data...")
    df = preprocess_data()
    
    # Use the lemmatized text for training
    X = df['lemmatized_text']
    y = df['label_num']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train the model
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    
    return accuracy

def predict_email(email_text):
    """Predict if an email is spam or ham"""
    global model, vectorizer
    
    if model is None or vectorizer is None:
        raise ValueError("Model not trained. Please train the model first.")
    
    # Preprocess the input text (simplified version)
    import re
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    
    # Download required NLTK data
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
    
    # Text preprocessing
    text = email_text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'[\r\n]', '', text)
    
    # Tokenize and lemmatize
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in lemmatized_tokens if token not in stop_words]
    
    # Join tokens back to text
    processed_text = ' '.join(filtered_tokens)
    
    # Vectorize and predict
    text_vectorized = vectorizer.transform([processed_text])
    prediction = model.predict(text_vectorized)[0]
    probability = model.predict_proba(text_vectorized)[0]
    
    return {
        'prediction': 'Spam' if prediction == 1 else 'Ham',
        'confidence': float(max(probability)),
        'spam_probability': float(probability[1]),
        'ham_probability': float(probability[0])
    }

@app.route('/')
def home():
    """Home page with a simple form"""
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Email Spam Classifier</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { background: #f5f5f5; padding: 20px; border-radius: 10px; }
            textarea { width: 100%; height: 200px; margin: 10px 0; padding: 10px; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background: #0056b3; }
            .result { margin-top: 20px; padding: 15px; border-radius: 5px; }
            .ham { background: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
            .spam { background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Email Spam Classifier</h1>
            <p>Enter an email text to classify it as spam or ham:</p>
            <form method="POST">
                <textarea name="email_text" placeholder="Enter email content here..."></textarea><br>
                <button type="submit">Classify Email</button>
            </form>
            
            {% if result %}
            <div class="result {{ 'spam' if result.prediction == 'Spam' else 'ham' }}">
                <h3>Classification Result:</h3>
                <p><strong>Prediction:</strong> {{ result.prediction }}</p>
                <p><strong>Confidence:</strong> {{ "%.2f"|format(result.confidence * 100) }}%</p>
                <p><strong>Spam Probability:</strong> {{ "%.2f"|format(result.spam_probability * 100) }}%</p>
                <p><strong>Ham Probability:</strong> {{ "%.2f"|format(result.ham_probability * 100) }}%</p>
            </div>
            {% endif %}
        </div>
    </body>
    </html>
    '''
    return render_template_string(html)

@app.route('/', methods=['POST'])
def classify_email():
    """Classify the submitted email"""
    email_text = request.form.get('email_text', '')
    
    if not email_text.strip():
        return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head><title>Error</title></head>
        <body>
            <h1>Error</h1>
            <p>Please enter some text to classify.</p>
            <a href="/">Go back</a>
        </body>
        </html>
        ''')
    
    try:
        result = predict_email(email_text)
        return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Email Spam Classifier</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                .container { background: #f5f5f5; padding: 20px; border-radius: 10px; }
                textarea { width: 100%; height: 200px; margin: 10px 0; padding: 10px; }
                button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
                button:hover { background: #0056b3; }
                .result { margin-top: 20px; padding: 15px; border-radius: 5px; }
                .ham { background: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
                .spam { background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Email Spam Classifier</h1>
                <p>Enter an email text to classify it as spam or ham:</p>
                <form method="POST">
                    <textarea name="email_text" placeholder="Enter email content here...">{{ email_text }}</textarea><br>
                    <button type="submit">Classify Email</button>
                </form>
                
                <div class="result {{ 'spam' if result.prediction == 'Spam' else 'ham' }}">
                    <h3>Classification Result:</h3>
                    <p><strong>Prediction:</strong> {{ result.prediction }}</p>
                    <p><strong>Confidence:</strong> {{ "%.2f"|format(result.confidence * 100) }}%</p>
                    <p><strong>Spam Probability:</strong> {{ "%.2f"|format(result.spam_probability * 100) }}%</p>
                    <p><strong>Ham Probability:</strong> {{ "%.2f"|format(result.ham_probability * 100) }}%</p>
                </div>
            </div>
        </body>
        </html>
        ''', email_text=email_text, result=result)
    
    except Exception as e:
        return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head><title>Error</title></head>
        <body>
            <h1>Error</h1>
            <p>An error occurred: {{ error }}</p>
            <a href="/">Go back</a>
        </body>
        </html>
        ''', error=str(e))

@app.route('/train')
def train():
    """Train the model endpoint"""
    try:
        accuracy = train_model()
        return jsonify({
            'status': 'success',
            'message': 'Model trained successfully',
            'accuracy': accuracy
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for prediction"""
    try:
        data = request.get_json()
        email_text = data.get('email_text', '')
        
        if not email_text:
            return jsonify({'error': 'Email text is required'}), 400
        
        result = predict_email(email_text)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Train the model when starting the app
    print("Training the email classification model...")
    train_model()
    
    print("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=8080)
