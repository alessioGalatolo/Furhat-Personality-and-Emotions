# code from https://github.com/thepanacealab/covid19_twitter/blob/master/COVID_19_dataset_Tutorial.ipynb
# with some modifications. Downloads and pre-processes the twitter covid dataset

import gzip
import shutil
import os
import wget
import csv
import linecache
from shutil import copyfile
import pandas as pd
import json
import tweepy
import math
import zipfile
import os.path as osp
from time import sleep
import argparse
from utils.parse_json_lite import preprocess_text
from utils.model_wrapper import PersonalityTransfer


parser = argparse.ArgumentParser()
parser.add_argument('--bearer-token',
                    dest='BEARER_TOKEN',
                    default='')
parser.add_argument('--access-token-key',
                    dest='ACCESS_TOKEN_KEY',
                    default='')
parser.add_argument('--access-token-secret-key',
                    dest='ACCESS_TOKEN_SECRET_KEY',
                    default='')
parser.add_argument('--consumer-key',
                    dest='CONSUMER_KEY',
                    default='')
parser.add_argument('--consumer-secret-key',
                    dest='CONSUMER_SECRET_KEY',
                    default='')
args = parser.parse_args()
dataset_URL = "https://github.com/thepanacealab/covid19_twitter/blob/master/dailies/2021-01-20/2021-01-20_clean-dataset.tsv.gz?raw=true"


# Downloads the dataset (compressed in a GZ format)
wget.download(dataset_URL, out='clean-dataset.tsv.gz')

# Unzips the dataset and gets the TSV dataset
with gzip.open('clean-dataset.tsv.gz', 'rb') as f_in:
    with open('clean-dataset.tsv', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

# Deletes the compressed GZ file
os.unlink("clean-dataset.tsv.gz")

# Gets all possible languages from the dataset
df = pd.read_csv('clean-dataset.tsv', sep="\t")

filtered_language = 'en'
if filtered_language == "":
    copyfile('clean-dataset.tsv', 'clean-dataset-filtered.tsv')

# If language specified, it will create another tsv file with the filtered records
else:
    filtered_tw = list()
    current_line = 1
    with open("clean-dataset.tsv") as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")

        if current_line == 1:
            filtered_tw.append(linecache.getline("clean-dataset.tsv", current_line))

            for line in tsvreader:
                if line[3] == filtered_language:
                    filtered_tw.append(linecache.getline("clean-dataset.tsv", current_line))
                current_line += 1

    print('\033[1mShowing first 5 tweets from the filtered dataset\033[0m')
    print(filtered_tw[1:(6 if len(filtered_tw) > 6 else len(filtered_tw))])

    with open('clean-dataset-filtered.tsv', 'w') as f_output:
        for item in filtered_tw:
            f_output.write(item)

inputfile = 'clean-dataset-filtered.tsv'
output_file = 'texts'
labels_file = 'labels'

# Authenticate
client = tweepy.Client(bearer_token=args.BEARER_TOKEN,
                       consumer_key=args.CONSUMER_KEY,
                       consumer_secret=args.CONSUMER_SECRET_KEY,
                       access_token=args.ACCESS_TOKEN_KEY,
                       access_token_secret=args.ACCESS_TOKEN_SECRET_KEY,
                       wait_on_rate_limit=True)

output_file_noformat = output_file.split(".", maxsplit=1)[0]
print(output_file)
output_file = '{}'.format(output_file)
output_file_short = '{}_short.json'.format(output_file_noformat)
compression = zipfile.ZIP_DEFLATED
ids = []

if '.tsv' in inputfile:
    inputfile_data = pd.read_csv(inputfile, sep='\t')
    print('tab seperated file, using \\t delimiter')
elif '.csv' in inputfile:
    inputfile_data = pd.read_csv(inputfile)
elif '.txt' in inputfile:
    inputfile_data = pd.read_csv(inputfile, sep='\n', header=None, names=['tweet_id'] )
    print(inputfile_data)

inputfile_data = inputfile_data.set_index('tweet_id')

ids = list(inputfile_data.index)
print('total ids: {}'.format(len(ids)))

start = 0
end = 100
limit = len(ids)
i = int(math.ceil(float(limit) / 100))

last_tweet = None
if osp.isfile(output_file) and osp.getsize(output_file) > 0:
    with open(output_file, 'rb') as f:
        # may be a large file, seeking without iterating
        f.seek(-2, os.SEEK_END)
        while f.read(1) != b'\n':
            f.seek(-2, os.SEEK_CUR)
        last_line = f.readline().decode()
    last_tweet = json.loads(last_line)
    start = ids.index(last_tweet['id'])
    end = start+100
    i = int(math.ceil(float(limit-start) / 100))

print('metadata collection complete')
print('Getting tweets and classifying them')

personality_classifier = PersonalityTransfer('utils.config')
text_labels = []
for go in range(i):
    print('currently getting {} - {}'.format(start, end))
    sleep(6)  # needed to prevent hitting API rate limit
    id_batch = ids[start:end]
    start += 100
    end += 100
    backOffCounter = 1
    tweets = client.get_tweets(id_batch)[0]
    for tweet in tweets:
        text = preprocess_text(tweet['data']['text'])
        try:
            label = personality_classifier.classify(text)
            label = 'ext' if label == 1 else 'int'
            text_labels.append({'text': text, 'label': label})
        except ValueError:
            print(f'Text "{text}" could not be classified, skipping it')

data = pd.DataFrame(text_labels)

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
