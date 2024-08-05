import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import tensorflow as tf
from utils.utils import load_yaml
from preprocessing.preprocessing import vectorize_sentences, vectorize_sentences_900dim

config = load_yaml('./config.yaml')
log_reg_filename = config['log_reg_model_filename']
LSTM_filename = config['LSTM_filename']
lightGBM_filename = config['lightGBM_filename']

def create_train_test_datasets(preprocessed_df: pd.DataFrame):

    X = vectorize_sentences(preprocessed_df['text'].values)
    print("Text vectorization finished")

    y = preprocessed_df['sentiment'].values

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=32
    )

    return X_train, X_test, y_train, y_test

def create_train_test_datasets_900dim(preprocessed_df: pd.DataFrame):

    X = vectorize_sentences_900dim(preprocessed_df['text'].values)
    print("Text vectorization finished")

    y = preprocessed_df['sentiment'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=32
    )

    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train, y_train):
    
    logRegModel = LogisticRegression(solver='liblinear', C=10.0, random_state=0)
    logRegModel.fit(X_train, y_train)
    print("Model training finished")

    joblib.dump(logRegModel, log_reg_filename)

def loading_saved_model(filename):
    return joblib.load(filename)

def predict_log_regression(X_test, y_test):    
    logRegModel = loading_saved_model(log_reg_filename)
    y_pred_logreg = logRegModel.predict(X_test)
    print("Prediction making finished")

    accuracy = accuracy_score(y_test, y_pred_logreg)
    print("Accuracy:", accuracy)

def train_LSTM(X_train, y_train):

    # Reshape X_train to have an additional dimension
    X_train_reshaped = np.expand_dims(X_train, axis=-1)

    max_length = 300

    Dense = tf.keras.layers.Dense
    Sequential = tf.keras.models.Sequential
    LSTM = tf.keras.layers.LSTM

    LSTMmodel = Sequential()
    LSTMmodel.add(LSTM(32, input_shape=(max_length, 1)))
    LSTMmodel.add(Dense(1, activation='sigmoid'))

    LSTMmodel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    LSTMmodel.summary()

    LSTMmodel.fit(X_train_reshaped, y_train, epochs=3, batch_size=32)

    print("Model training finished")

    LSTMmodel.save(LSTM_filename)

def predict_LSTM(X_test, y_test):

    LSTMmodel = tf.keras.models.load_model(LSTM_filename)
    
    loss, accuracy = LSTMmodel.evaluate(X_test, y_test)
    
    print(f'Test Accuracy: {accuracy:.4f}')


def train_LGBM_model(X_train, y_train):

    train_data = lgb.Dataset(X_train, label=y_train)

    params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'verbose': 2
    }

    lgbm_model = lgb.train(params, train_data, num_boost_round=100)

    print("Model training finished")

    joblib.dump(lgbm_model, lightGBM_filename)


def predict_LGBM_model(X_test, y_test):

    lgbm_model = loading_saved_model(lightGBM_filename)
    y_pred = lgbm_model.predict(X_test)
    y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred]
    print("Prediction making finished")

    accuracy = accuracy_score(y_test, y_pred_binary)
    print("Accuracy:", accuracy)
    

    