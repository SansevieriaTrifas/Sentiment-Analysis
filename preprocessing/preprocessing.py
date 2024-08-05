import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from gensim.models import KeyedVectors
import operator
import re
import nltk
from utils.utils import load_yaml

config = load_yaml('./config.yaml')

def build_vocabulary(sentences, verbose =  True):
    """
    :param sentences: list of lists of words
    :return: dictionary of words and their count
    """
    counts = {}
    for sentence in tqdm(sentences, disable = (not verbose)):
        for word in sentence:
            try:
                counts[word] += 1
            except KeyError:
                counts[word] = 1
    return counts

def load_word2vec_model():
    word2vec_file = config['word2vec_file']
    word2vec = KeyedVectors.load_word2vec_format(word2vec_file, binary=True)
    return word2vec

model = load_word2vec_model()
 
def check_coverage(vocabulary):
    have_embed = {}
    no_embed = {}
    have_embed_count = 0
    no_embed_count = 0
    for word in tqdm(vocabulary):
        try:
            have_embed[word] = model[word]
            have_embed_count += vocabulary[word]
        except:

            no_embed[word] = vocabulary[word]
            no_embed_count += vocabulary[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(have_embed) / len(vocabulary)))
    print('Found embeddings for  {:.2%} of all text'.format(have_embed_count / (have_embed_count + no_embed_count)))
    sorted_no_embed = sorted(no_embed.items(), key=operator.itemgetter(1))[::-1]

    return sorted_no_embed

def remove_mentions(text):
    # Removing hashtags and mentions
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'@\w+', '', text)

    # Removing URLs
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www\S+', '', text)
    text = re.sub(r'\S+\.com', '', text)
    return text

def clean_text(x):
    x = str(x)
    for punct in "/-'":
        x = x.replace(punct, ' ')
    for punct in '&':
        x = x.replace(punct, f' {punct} ')
    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
        x = x.replace(punct, '')
    return x

#numbers greater than 9 are replaced by ## 
def clean_numbers(x):

    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x

def _get_mispell():
    mispell_dict = config['mispell_dict']

    # Convert keys to Unicode strings
    mispell_dict = {key: value for key, value in mispell_dict.items()}
    mispell_re = re.compile('(%s)' % '|'.join(map(re.escape, mispell_dict.keys())))
    return mispell_dict, mispell_re


def replace_typical_misspell(text):

    mispellings, mispellings_re = _get_mispell()

    if isinstance(text, bytes):
        text = text.decode('latin')  

    def replace(match):
        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)

def remove_selected_stop_words(text):
    selected_stop_words = config['selected_stop_words']
    words = text.split()
    processed_words = [word for word in words if word not in selected_stop_words]
    processed_text = ' '.join(processed_words)
    return processed_text

# Function to compute embedding vector for a sentence
def compute_average_vector_900dim(sentence):
    vectors = []
    vector_size = model.vector_size
    avg_vector = np.zeros(vector_size)
    max_vector = np.zeros(vector_size)
    min_vector = np.zeros(vector_size) 
    sentence = nltk.word_tokenize(sentence)
    for word in sentence:
        if word in model:
            vectors.append(model[word])

    if len(vectors) == 0:
        # Handle the case when no valid words are found in the sentence
        final_vector = np.concatenate([avg_vector, max_vector, min_vector])
        return final_vector

    avg_vector = np.mean(vectors, axis=0)
    max_vector = np.max(vectors, axis=0)
    min_vector = np.min(vectors, axis=0)
    final_vector = np.concatenate([avg_vector, max_vector, min_vector])
    return final_vector

# Function to compute embedding vector for a sentence
def compute_average_vector(sentence):
    vectors = []
    vector_size = model.vector_size
    avg_vector = np.zeros(vector_size)
    sentence = nltk.word_tokenize(sentence)
    for word in sentence:
        if word in model:
            vectors.append(model[word])

    if len(vectors) == 0:
        # Handle the case when no valid words are found in the sentence
        return avg_vector

    avg_vector = np.mean(vectors, axis=0)
    return avg_vector


def save_preprocessed_df(preprocessed_df : pd.DataFrame):
    preprocessed_df_filename = config['preprocessed_df_filename']     
    preprocessed_df.to_pickle(preprocessed_df_filename)

def preprocess_text(dataframe: pd.DataFrame):

    dataframe["text"] = dataframe["text"].progress_apply(lambda x: remove_mentions(x))
    dataframe["text"] = dataframe["text"].progress_apply(lambda x: clean_text(x))
    dataframe["text"] = dataframe["text"].progress_apply(lambda x: clean_numbers(x))
    dataframe["text"] = dataframe["text"].progress_apply(lambda x: replace_typical_misspell(x))
    dataframe["text"] = dataframe["text"].progress_apply(lambda x: remove_selected_stop_words(x))

    return dataframe


def vectorize_sentences(sentences):
    return np.array([compute_average_vector(sentence) for sentence in tqdm(sentences)])


def vectorize_sentences_900dim(sentences):
    return np.array([compute_average_vector_900dim(sentence) for sentence in tqdm(sentences)])