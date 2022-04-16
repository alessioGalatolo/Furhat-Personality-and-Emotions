import shutil
import os
import wget
import pandas as pd
from utils.parse_json_lite import preprocess_text
from utils.model_wrapper import PersonalityTransfer

dataset_URL = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"

# Downloads the dataset (compressed in a GZ format)
wget.download(dataset_URL, out='twitter_dataset.zip')

# Unzips the dataset and gets the csv dataset
shutil.unpack_archive('twitter_dataset.zip', './twitter')

# Deletes the compressed GZ file
os.unlink("twitter_dataset.zip")

# Gets all possible languages from the dataset
data = pd.read_csv('./twitter/training.1600000.processed.noemoticon.csv',
                   encoding='Windows-1252',
                   names=['polarity', 'id', 'date', 'query', 'user', 'text'])

personality_classifier = PersonalityTransfer('utils.config')
data['text'] = data['text'].apply(preprocess_text)


def classify(x):
    try:
        label = personality_classifier.classify(x)
        label = 'ext' if label == 1 else 'int'
    except ValueError:
        label = 'invalid'
    return label


data['label'] = data['text'].apply(classify)
data = data[data['label'] != 'invalid']

train = data.sample(frac=0.8)
val_test = data.drop(train.index)
val = val_test.sample(frac=0.5)
test = val_test.drop(val.index)

train['text'].to_csv("train.txt", index=False, header=False)
val['text'].to_csv("dev.txt", index=False, header=False)
test['text'].to_csv("test.txt", index=False, header=False)
train['label'].to_csv("train.label", index=False, header=False)
val['label'].to_csv("dev.label", index=False, header=False)
test['label'].to_csv("test.label", index=False, header=False)
