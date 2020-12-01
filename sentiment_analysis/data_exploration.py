import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import string

from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.corpus import stopwords

import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Input, LSTM, Conv1D, MaxPool1D, Bidirectional, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# TODO: Include WebScraping to increase the data

# load the sentiment data
stock_df = pd.read_csv('../data/stock_sentiment.csv')

# print(stock_df.info())

# print(stock_df.isnull().sum())

# unique elements in the sentiment column
# we have an unbalanced dataset which might cause problems when training our model
no_unique_elements = stock_df['Sentiment'].nunique();


# sns.countplot(stock_df['Sentiment'])
# plt.show()


# Data Cleaning - Removing Punctuation
def remove_punc(text):
  return ''.join([char for char in text if char not in string.punctuation])


stock_df['Text without Punctuation'] = stock_df['Text'].apply(remove_punc)

# Data Cleaning - Removing Stopwords
stopwords_english = stopwords.words('english')
stopwords_english.extend(
  ['from', 'subject', 're', 'edu', 'use', 'will', 'aap', 'co', 'day', 'user', 'stock', 'today', 'week', 'year',
   'https'])

print(stopwords_english)


# returns String[]
def preprocess(text):
  result = []
  for token in gensim.utils.simple_preprocess(text):
    if token not in stopwords_english and len(token) > 1:
      result.append(token)
  return result


# this column contains string arrays
stock_df['Text without Punctuation and Stopwords'] = stock_df['Text without Punctuation'].apply(preprocess)

stock_df['Preprocessed Text'] = stock_df['Text without Punctuation and Stopwords'].apply(lambda x: " ".join(x))


# print(stock_df)

# pass in the column name as a String i.e. 'Preprocessed Text'
def plot_positive_sentiment(df, column):
  plt.figure(figsize=(10, 10))
  word_cloud = WordCloud(max_words=1000, width=1600, height=800).generate(" ".join(df[df['Sentiment'] == 1][column]))
  plt.imshow(word_cloud)
  plt.show()


def plot_negative_sentiment(df, column):
  plt.figure(figsize=(10, 10))
  word_cloud = WordCloud(max_words=1000, width=1600, height=800).generate(" ".join(df[df['Sentiment'] == 0][column]))
  plt.imshow(word_cloud)
  plt.show()


# get the max number of words in the entire dataset to be used in training the data
maxlen = -1
for text in stock_df['Preprocessed Text']:
  tokens = nltk.word_tokenize(text)
  if len(tokens) > maxlen:
    maxlen = len(tokens)

print('Max number of words in any document is', maxlen)

tweets_length_array = [len(nltk.word_tokenize(x)) for x in stock_df['Preprocessed Text']]
print(tweets_length_array)


def plot_histogram_tweet_length(array):
  fig = px.histogram(x=array, nbins=50)
  fig.show()


# Tokenising the data - converting to int[]
# obtain the total words present in the dataset
list_of_words = []
for i in stock_df['Text without Punctuation and Stopwords']:
  for j in i:
    list_of_words.append(j)

print("total number of words in my whole dataset:", len(list_of_words))

list_of_unique_words = list(set(list_of_words))
total_words = len(list_of_unique_words)
print("total number of unique words in my whole dataset:", total_words)

# Split Data into Test and Train
X = stock_df['Text without Punctuation and Stopwords']
y = stock_df['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Create a tokenizer to tokenize the words and create sequences of tokenized words
tokenizer = Tokenizer(num_words=total_words)
tokenizer.fit_on_texts(X_train)

# Training Data - this is the tokenized words data
train_sequences = tokenizer.texts_to_sequences(X_train)

# Testing Data
test_sequences = tokenizer.texts_to_sequences(X_test)

# Adding Padding so all our tokenized arrays have the same lengths, we will add 0's prefix
padded_train = pad_sequences(train_sequences, maxlen=29)
padded_test = pad_sequences(test_sequences, maxlen=29)

# for i, doc in enumerate(padded_train[:3]):
#   print('The padded encoding for document {} is {}'.format(i + 1, doc))

# Convert y_train and y_test into a categorical 2D representation
y_train_cat = to_categorical(y_train, 2)
y_test_cat = to_categorical(y_test, 2)

# Building Model in Sequential manner
model = Sequential()
model.add(Embedding(total_words, output_dim=512))
model.add(LSTM(256))
model.add(Dense(128, activation='relu'))  # relu = rectify linear unit
model.add(Dropout(0.3))  # drops some random neurons to avoid overfitting
# output layer
model.add(Dense(2, activation='softmax'))  # softmax is a nice activation function for binary classification

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
# print(model.summary())

# input -> padded_train
# output -> y_train_cat
model.fit(padded_train, y_train_cat, batch_size=32, validation_split=0.2, epochs=2)

# predictions
pred = model.predict(padded_test)
print(pred)

# convert the predictions matrix into 0's and 1's
# argmax finds the arguement that gives the max value & used to find the class with the highest probability (pred)
prediction = []
for val in pred:
  prediction.append(np.argmax(val))

# do the same on the original dataset to print out the confusion matrix
original = []
for val in y_test_cat:
  original.append(np.argmax(val))

# accuracy score
accuracy = accuracy_score(original, prediction)
print(accuracy)

# Confusion Matrix
cm = confusion_matrix(original, prediction)
sns.heatmap(cm, annot=True)
plt.show()
