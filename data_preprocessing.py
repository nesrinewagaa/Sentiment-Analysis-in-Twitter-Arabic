# =========================
# data_preprocessing.py
# Arabic Tweet Sentiment Analysis - Preprocessing
# =========================

# 1. Import Libraries
import re
import string
import pandas as pd
import numpy as np
from nltk.stem.isri import ISRIStemmer
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Download stopwords (if needed later)
nltk.download('stopwords')
from nltk.corpus import stopwords

# =========================
# 2. Load Dataset
# =========================
train_path = 'SemEval2017-train.txt'
test_path  = 'SemEval2017-test.txt'

# Load datasets
df_train = pd.read_table(train_path, usecols=[1,2], encoding='utf-8', names=['sentiment', 'tweet'])
df_test  = pd.read_table(test_path, usecols=[1,2], encoding='utf-8', names=['sentiment', 'tweet'])

# Combine datasets
data = pd.concat([df_train, df_test], axis=0, ignore_index=True)
data = data.dropna()  # Drop missing values

# =========================
# 3. Text Preprocessing Functions
# =========================

# a) Remove hashtags
def remove_hashtags(text):
    return ' '.join(word for word in text.split() if not word.startswith('#'))

# b) Remove punctuations (Arabic + English)
arabic_punctuations = '''؛><_)(*&^%[]ـ:/،"؟,.'}{~¦+|!“…”–ـ×÷`'''
english_punctuations = string.punctuation
punctuations_list = arabic_punctuations + english_punctuations

def remove_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)

# c) Normalize Arabic text
def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text

# d) Remove repeating characters
def remove_repeating_char(text):
    return re.sub(r'(.)\1+', r'\1', text)

# e) Remove Arabic diacritics
def dNoise(text):
    noise = re.compile(r''' ّ | َ | ً | ُ | ٌ | ِ | ٍ | ْ | ـ''', re.VERBOSE)
    return re.sub(noise, '', text)

# f) Full preprocessing pipeline
stemmer = ISRIStemmer()
def processDocument(doc, stemmer):
    doc = re.sub(r'@[^\s]+', ' ', doc)            # Remove mentions
    doc = re.sub(r'\n', ' ', doc)                # Remove line breaks
    doc = re.sub(r'[a-zA-Z]', '', doc)           # Remove English letters
    doc = re.sub(r'\d', '', doc)                 # Remove digits
    doc = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))',' ', doc)  # Remove URLs
    doc = remove_hashtags(doc)
    doc = remove_punctuations(doc)
    doc = normalize_arabic(doc)
    doc = remove_repeating_char(doc)
    doc = stemmer.stem(doc)
    doc = dNoise(doc)
    doc = " ".join([word for word in doc.split() if len(word) > 2])  # Remove short words
    return doc

# Apply preprocessing
data['tweet'] = data['tweet'].apply(lambda x: processDocument(x, stemmer))

# =========================
# 4. Encode Sentiment Labels
# =========================
def encode_sentiment(sent):
    mapping = {"positive": 1, "negative": 0, "neutral": 2}
    return mapping.get(sent, 2)

data['sentiment'] = data['sentiment'].apply(encode_sentiment)

# One-hot encode labels
labels = tf.keras.utils.to_categorical(np.asarray(data['sentiment']))

# =========================
# 5. Tokenization & Padding
# =========================
X = data['tweet']
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, padding='post', maxlen=500)

print("Vocabulary size:", len(tokenizer.word_index))

# =========================
# 6. Train/Test Split
# =========================
X_train, X_test, Y_train, Y_test = train_test_split(
    X_pad, labels, test_size=0.3, random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# =========================
# 7. Save Preprocessed Data
# =========================
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('Y_train.npy', Y_train)
np.save('Y_test.npy', Y_test)
np.save('tokenizer.npy', tokenizer.word_index)  # Optional
print("Preprocessed data saved successfully.")

