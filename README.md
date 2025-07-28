# Email Spam Classifier

A machine learning application that classifies emails as spam or ham (legitimate) using Natural Language Processing (NLP) and a Naive Bayes classifier.

## Features

- **Text Preprocessing**: HTML tag removal, special character cleaning, tokenization, lemmatization, and stopword removal
- **Machine Learning Model**: Multinomial Naive Bayes classifier with TF-IDF vectorization
- **Web Interface**: Flask-based web application with a user-friendly interface
- **API Endpoints**: RESTful API for programmatic access
- **High Accuracy**: Achieves ~95% accuracy on the test dataset

## Project Structure

```
Email_class/
├── app.py                 # Flask web application
├── data_preprocessing.py  # Data preprocessing pipeline
├── requirements.txt       # Python dependencies
├── spam_ham_dataset.csv  # Training dataset
└── README.md            # This file
```

## Installation

1. Clone or download the project files
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Web Application

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:8080
```

3. Enter email text in the text area and click "Classify Email" to get predictions.

### API Usage

The application provides RESTful API endpoints:

#### Train the model:
```bash
curl http://localhost:8080/train
```

#### Predict email classification:
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"email_text": "Your email content here"}'
```

### Data Preprocessing

To run the data preprocessing pipeline separately:

```bash
python data_preprocessing.py
```

This will:
- Load the dataset
- Clean and preprocess the text
- Tokenize and lemmatize words
- Remove stopwords
- Display the processed dataset information

## Model Performance

The trained model achieves:
- **Accuracy**: ~95%
- **Precision**: 96% for Ham, 91% for Spam
- **Recall**: 96% for Ham, 91% for Spam
- **F1-Score**: 96% for Ham, 91% for Spam

## Technical Details

### Preprocessing Pipeline

1. **Text Cleaning**: Convert to lowercase, remove HTML tags, special characters, and newlines
2. **Tokenization**: Split text into individual words
3. **Lemmatization**: Reduce words to their base form (e.g., "running" → "run")
4. **Stopword Removal**: Remove common words that don't add meaning
5. **TF-IDF Vectorization**: Convert text to numerical features

### Model Architecture

- **Algorithm**: Multinomial Naive Bayes
- **Feature Extraction**: TF-IDF with 5000 maximum features
- **Training/Test Split**: 80/20 split with random state 42

## Dataset

The model is trained on a spam/ham email dataset with:
- 5,171 total emails
- Binary classification (spam = 1, ham = 0)
- Text content for feature extraction

## Dependencies

- **numpy<2.0.0**: Numerical computing (constrained for compatibility)
- **pandas>=1.5.0**: Data manipulation
- **nltk>=3.8**: Natural language processing
- **scikit-learn>=1.0.0**: Machine learning algorithms
- **flask>=2.0.0**: Web framework
- **spacy>=3.0.0**: Advanced NLP library

## Troubleshooting

### NumPy Compatibility Issues

If you encounter NumPy compatibility errors, the requirements.txt file already specifies `numpy<2.0.0` to avoid conflicts with other packages.

### Port Issues

If port 8080 is already in use, you can modify the port in `app.py`:

```python
app.run(debug=True, host='0.0.0.0', port=YOUR_PORT)
```

### NLTK Data

The application automatically downloads required NLTK data on first run. If you encounter NLTK-related errors, you can manually download the data:

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
```

## License

This project is for educational purposes. The dataset and code are provided as-is.
