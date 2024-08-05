import pandas as pd

def load_data() -> pd.DataFrame:
                        
    """
    Load dataset from csv from raw_files folder

    Returns:
       pandas DataFrame
    """

    dataframe = pd.read_csv("./data/raw_files/tweetsKaggle.csv", encoding="latin")
    dataframe.columns = ['sentiment', 'id', 'date', 'flag', 'user', 'text']
    dataframe['sentiment'] = pd.Categorical(dataframe['sentiment'], categories=[0, 4], ordered=True).codes

    return dataframe