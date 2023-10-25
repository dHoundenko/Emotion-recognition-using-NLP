import re
import inflect
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.read().splitlines()
    return lines

def mapping(key_list):
    emotion_map = {
        '0': "anger",
        '1': "joy",
        '2': "optimism",
        '3': "sadness",
    }

    emotions = [emotion_map[num] for num in key_list]

    return emotions

def specific_case(text):
    result = re.sub(r'(&gt;){3}', 'is better than', text)
    result = result.replace("szn", "season")
    result = re.sub(r'&[gl]t;?', '', result)
    result = result.replace("ó", "o")
    result = result.replace("ñ", "n")
    result = result.replace("é", "e")
    return result

def normalize_repeated_characters(text):
    # Replace 3 or more consecutive characters with just one
    return re.sub(r'(.)\1{2,}', r'\1', text)

def remove_user_mentions(text):
    return re.sub(r'@(\w+)', r'\1', text)

def process_more_sign(text):
    result = re.sub(r'\s*user \+', 'user', text)
    result = re.sub(r'#\++', '', result)
    result = re.sub(r'(?<=\d)\+', ' more ', result)
    result = re.sub(r'(?<=\s)\+', ' plus ', result)
    result = re.sub(r'\+1', ' plus one ', result)
    return result

def process_dollar(text):
    result = re.sub(r'\${2,}', 'cash', text)
    pattern = r'\$(\d+(?:\.\d{2})?)'
    result = re.sub(pattern, lambda match: match.group(1) + ' dollars ', result)
    result = re.sub(r'\$*', '', result)
    return result

def process_euro(text):
    pattern = r'\€(\d+(?:\.\d{2})?)'
    result = re.sub(pattern, lambda match: match.group(1) + ' euros ', text)
    return result

def process_pounds(text):
    pattern = r'\£(\d+(?:\.\d{2})?)'
    result = re.sub(pattern, lambda match: match.group(1) + ' pounds ', text)
    return result

def process_percent(text):
    pattern = r'(?:\s+|\d+(?:\.\d{0,2})?)%'
    result = re.sub(pattern, lambda match: match.group(0).replace('%', ' percent '), text)
    result = re.sub(r'%', '', result)
    return result

def process_equal(text):
    result = re.sub(r'=', ' equals ', text)
    return result

def process_at(text):
    result = re.sub(r'(?<=\s)@(?=\s)', ' at ', text)
    return result

def remove_newlines(text):
    return re.sub(r'\\n', ' ', text)

def process_amp(text):
    return re.sub(r'&amp;?', ' and ', text)

def process_hyphen(text):
    return re.sub(r'(\d+)\s*-\s*(\d+)', r'\1 to \2', text)

def replace_numbers_with_words(text):
    p = inflect.engine()

    number_pattern = r'(\d+\.\d+|\d+)'

    numbers = re.findall(number_pattern, text)

    for number in numbers:
        word_representation = p.number_to_words(number)
        text = re.sub(re.escape(number), word_representation, text)

    return text

def clear_special_characters(text):
    return re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

def lowercase_text(text):
    return text.lower()

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    return ' '.join(lemmatizer.lemmatize(word, wordnet.VERB) for word in tokens)

def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    tokens = nltk.word_tokenize(text)
    return ' '.join(word for word in tokens if word.lower() not in stop_words)

def process_text(text):
    text = specific_case(text)
    text = remove_user_mentions(text)
    text = process_more_sign(text)
    text = process_dollar(text)
    text = process_euro(text)
    text = process_pounds(text)
    text = process_percent(text)
    text = process_hyphen(text)
    text = process_equal(text)
    text = process_at(text)
    text = remove_newlines(text)
    text = process_amp(text)
    text = replace_numbers_with_words(text)
    text = normalize_repeated_characters(text)
    text = clear_special_characters(text)
    text = lowercase_text(text)
    text = lemmatize_text(text)
    text = remove_stopwords(text)
    return text

def create_df(keys_file, values_file):
    keys = read_file(keys_file)
    keys = mapping(keys)    
    values = read_file(values_file)

    processed_values = [process_text(value) for value in values]

    data_dict = {
        "text": processed_values,
        "emotions": keys
    }

    df = pd.DataFrame(data_dict)

    return df

def get_data(file):
    return pd.read_pickle(file)

def combine_df(df1, df2):
    return pd.concat([df1, df2], ignore_index=True)

# function to get the dataset as a panda dataframe
def get_dataset():
    keys_file = "DatasetsInUse/emotion_tweets_2020/train_labels.txt"
    values_file = "DatasetsInUse/emotion_tweets_2020/train_text.txt"

    df1 = create_df(keys_file, values_file)

    file = "DatasetsInUse/emotion/merged_training.pkl"
    df2 = get_data(file)

    df = combine_df(df1, df2)

    return df
