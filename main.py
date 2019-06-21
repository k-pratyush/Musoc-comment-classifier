### Musoc 2019
### Machine Learning Model for Classification of toxic comments

### @author: Pratyush Kerhalkar

import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

logging.basicConfig(filename = "classifier_debug.log", level = logging.DEBUG, format = '%(asctime)s:%(levelname)s:%(message)s')

logging.debug("loading data...")
train_data = pd.read_csv("Datasets/train.csv")
test_data = pd.read_csv("Datasets/test.csv")
test_labels = pd.read_csv("Datasets/test_labels.csv")
logging.debug("data loaded")

test_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

train_text = train_data["comment_text"]
test_text = test_data["comment_text"]

text = pd.concat([train_text, test_text])

logging.debug("creating tfidf vectorizer for input dataset")
word_vectorizer = TfidfVectorizer(
    min_df = 1,
    stop_words= 'english')

word_vectorizer.fit(text)

train_features = word_vectorizer.transform(train_text)
test_features = word_vectorizer.transform(test_text)

logging.debug("Creating model")
# Sample Logistic Regression model for "toxic" comment feature
model = LogisticRegression(C = 0.1, solver='sag')
model.fit(train_features, train_data["toxic"])
predicted_value = model.predict(train_features)
logging.debug("Model created")

example = ["Enter a comment"]
example_text = pd.DataFrame(example)
op = word_vectorizer.transform(example_text[0])
example_predict = model.predict(op)
print(example_predict)
if(example_predict):
	print(example, "is a toxic comment")
else:
	print(example, "is not a toxic comment")

# Accuracy Calculation for sample logistic regression model
true_value = train_data["toxic"]
true_value_numpy = true_value.to_numpy()
total = len(true_value_numpy)
count = 0

for i in range(len(true_value_numpy)):
    if true_value_numpy[i] == predicted_value[i]:
        count += 1

accuracy = float(count/total)*100
print("Model accuracy obtained: ", accuracy)
logging.debug("Model accuracy obtained: " + str(accuracy))
# Accuracy obtained: 93.52%


# Logistic regression model creation
output_logistic_regression = pd.DataFrame.from_dict({'id': test_data['id']})

for feature in test_classes:
    model = LogisticRegression(C=0.1, solver='sag')
    model.fit(train_features, train_data[feature])
    output_logistic_regression[feature] = model.predict_proba(test_features)[:,1]
    
logging.debug("Completed execution")