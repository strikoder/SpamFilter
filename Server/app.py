import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
#TF
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout
import pickle

ps = PorterStemmer()

with open('K:\Github\DS & ML & DL\Projects\Machine & Deep learning projects\(Kaggle)SMSSpamClassifier\Model\Vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('K:\Github\DS & ML & DL\Projects\Machine & Deep learning projects\(Kaggle)SMSSpamClassifier\Model\model.pkl', 'rb') as f:
    model = pickle.load(f)


def transform_text(text):
    # Convert the text to lowercase and tokenize it
    tokens = nltk.word_tokenize(text.lower())

    # Remove non-alphanumeric tokens (%^# )
    tokens = [t for t in tokens if t.isalnum()]

    # Remove stopwords and punctuation (I, how, u, are, is)
    stopwords_set = set(stopwords.words('english'))
    punctuation_set = set(string.punctuation)
    tokens = [t for t in tokens if t not in stopwords_set and t not in punctuation_set]

    # Stemming the remaining tokens using PorterStemmer (loving=love)
    ps = PorterStemmer()
    tokens = [ps.stem(t) for t in tokens]

    # Join the tokens back into a string and return it
    return " ".join(tokens)


st.title("SMS Spam Classifier")

input_sms = st.text_area("Enter the SMS message")

if st.button("Predict"):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms]).toarray()
    result = model.predict(vector_input)[0]
    if result == 1:
        st.header("Spam")
    else:
        st.header("Ham")


