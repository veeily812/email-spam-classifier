import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Global variable to store the dataframe
df = None

#load dataset
def load():
    global df
    df = pd.read_csv("spam_ham_dataset.csv")
    return df

#remove html tag and special characters
def remove_char():
    global df
    if df is None:
        df = load()
    df['text'] = df['text'].str.lower()
    df['text'] = df['text'].apply(lambda x: re.sub(r'<.*?>', '', x))
    df['text'] = df['text'].apply(lambda x: re.sub(r'[^a-z0-9\s]', '', x))
    df['text'] = df['text'].apply(lambda x: re.sub(r'[\r\n]', '', x))
    return df

#tokenize words
def tokenize_text():
    global df
    if df is None:
        df = load()
    df['tokens'] = df['text'].apply(word_tokenize)
    return df

def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

def apply_lemmatization():
    global df
    if df is None:
        df = load()
    df['lemmatized_tokens'] = df['tokens'].apply(lemmatize_tokens)
    return df

def join_tokens(tokens):
    return ' '.join(tokens)

def create_lemmatized_text():
    global df
    if df is None:
        df = load()
    df['lemmatized_text'] = df['lemmatized_tokens'].apply(join_tokens)
    return df

def remove_stopwords(tokens):
    return [token for token in tokens if token not in stop_words]

def apply_stopword_removal():
    global df
    if df is None:
        df = load()
    df['tokens_no_stopwords'] = df['lemmatized_tokens'].apply(remove_stopwords)
    return df

def preprocess_data():
    """Main function to run all preprocessing steps"""
    global df
    df = load()
    df = remove_char()
    df = tokenize_text()
    df = apply_lemmatization()
    df = create_lemmatized_text()
    df = apply_stopword_removal()
    return df

if __name__ == "__main__":
    # Run the preprocessing pipeline
    processed_df = preprocess_data()
    print("Data preprocessing completed!")
    print(f"Dataset shape: {processed_df.shape}")
    print(f"Columns: {list(processed_df.columns)}")
