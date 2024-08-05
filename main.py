import pandas as pd
from data.data_loading import load_data
from preprocessing.preprocessing import build_vocabulary, check_coverage, save_preprocessed_df, preprocess_text
from models.training import create_train_test_datasets , predict_LSTM, train_LSTM #train_LGBM_model, predict_LGBM_model #train_logistic_regression, predict_log_regression

def main():
    """
    The main entry point of the application.
    """
    #dataframe = load_data()
    #print(dataframe.head())

    #tweets = dataframe["text"].progress_apply(lambda x: x.split()).values

    #vocab_dictionary = build_vocabulary(tweets)
    #check_coverage(vocab_dictionary)

    #preprocessed_df = preprocess_text(dataframe)
    #save_preprocessed_df(preprocessed_df)

    #sentences = preprocessed_df["text"].progress_apply(lambda x: x.split())
    #vocab = build_vocabulary(sentences)
    #check_coverage(vocab)

    loaded_dataframe = pd.read_pickle('./preprocessing/saved_dataframe/preprocessed_df.pkl')
    X_train, X_test, y_train, y_test = create_train_test_datasets(loaded_dataframe)

    #train_logistic_regression(X_train, y_train)
    #predict_log_regression(X_test, y_test)

    #train_LGBM_model(X_train, y_train)
    #predict_LGBM_model(X_test, y_test)

    train_LSTM(X_train, y_train)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    predict_LSTM(X_test, y_test)

if __name__ == "__main__":
    main()