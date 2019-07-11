### Musoc 2019
### Machine Learning Model for Classification of toxic comments

### @author: Pratyush Kerhalkar

import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
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

logging.debug("creating count vectorizer for input dataset")
count_word_vectorizer = CountVectorizer(
    min_df = 1,
    stop_words= 'english')


logging.debug("creating tfidf vectorizer for input dataset")
tfidf_word_vectorizer = TfidfVectorizer(
    min_df = 1,
    stop_words= 'english')


count_word_vectorizer.fit(text)
tfidf_word_vectorizer.fit(text)

train_features_cv = count_word_vectorizer.transform(train_text)
test_features_cv = count_word_vectorizer.transform(test_text)


train_features = tfidf_word_vectorizer.transform(train_text)
test_features = tfidf_word_vectorizer.transform(test_text)


#logging.debug("Creating naive bayes model with count vectorizer")
# Sample NB model for "toxic" comment feature
#toxic_model_nb = MultinomialNB()
#toxic_model_nb.fit(train_features_cv, train_data['toxic'])
#predicted_value_nb = toxic_model_nb.predict(train_features_cv)


'''
# Accuracy Calculation for sample NB model
true_value = train_data["toxic"]
true_value_numpy = true_value.to_numpy()
total = len(true_value_numpy)
count = 0

for i in range(len(true_value_numpy)):
    if true_value_numpy[i] == predicted_value_nb[i]:
        count += 1

accuracy = float(count/total)*100
print("Model accuracy for NB obtained: ", accuracy)
logging.debug("Model accuracy for NB obtained: " + str(accuracy))

'''


logging.debug("Creating logistic regression model with tfidf vectorizer")
# Sample Logistic Regression model for "toxic" comment feature
model = LogisticRegression(C = 0.1, solver='sag')
model.fit(train_features, train_data["toxic"])
predicted_value = model.predict(train_features)
#predicted_output = model.predict_proba(test_features)
logging.debug("Model created")


# Accuracy Calculation for sample logistic regression model
true_value = train_data["toxic"]
true_value_numpy = true_value.to_numpy()
total = len(true_value_numpy)
count = 0

for i in range(len(true_value_numpy)):
    if true_value_numpy[i] == predicted_value[i]:
        count += 1

accuracy = float(count/total)*100
print("Model accuracy for logistic regression obtained: ", accuracy)
logging.debug("Model accuracy for logistic regression obtained: " + str(accuracy))


'''
example = ['Enter Text']
example_text = pd.DataFrame(example)
op = count_word_vectorizer.transform(example_text[0])
example_predict = toxic_model_nb.predict(op)
print(example_predict)


if(example_predict):
	print(example, "is a toxic comment")
else:
	print(example, "is not a toxic comment")
'''


# Logistic regression model creation
output_logistic_regression = pd.DataFrame.from_dict({'id': test_data['id']})

for feature in test_classes:
    model = LogisticRegression(C=0.1, solver='sag')
    model.fit(train_features, train_data[feature])
    output_logistic_regression[feature] = model.predict_proba(test_features)[:,1]

output_logistic_regression.to_csv("Datasets/lg_results.csv")
print(output_logistic_regression)

output_nb_model = pd.DataFrame.from_dict({'id': test_data['id']})

for feature in test_classes:
    model_nb = MultinomialNB()
    model_nb.fit(train_features, train_data[feature])
    output_nb_model[feature] = model.predict_proba(test_features_cv)[:,1]

output_nb_model.to_csv("Datasets/nb_results.csv")
print(output_nb_model)
logging.debug("Completed execution")
