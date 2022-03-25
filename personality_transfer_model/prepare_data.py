# Copyright 2018 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Downloads data.
"""
import argparse
from collections import defaultdict
from os import makedirs, mkdir, path, rename, listdir
from re import findall
import string
import pandas as pd
from tqdm import tqdm
import texar.torch as tx


def prepare_yelp(**kwargs):
    """Downloads data.
    """
    rename(r"./yelp/sentiment.*", r"./yelp/*")


def prepare_ear():
    ...


def prepare_essays(base_path, max_length, text_file, label_file, vocab_file, interactive, expand_essays=True):
    traits_labels = ['cOPN', 'cCON', 'cEXT', 'cAGR', 'cNEU']
    with open(f"{base_path}/essays", "r") as essays:
        data = pd.read_csv(essays)
    if expand_essays:
        print('Expanding the essays into single sentences, this will probably take a long time...')
        unexpanded_data = data
        translation_table = defaultdict(lambda: ' ', {ord(letter): letter for letter in string.ascii_letters})
        translation_table[ord('.')] = ' . '
        translation_table[ord('?')] = ' ? '
        translation_table[ord(',')] = ' , '
        translation_table[ord('!')] = ' ! '

        data = pd.DataFrame()
        with open(text_file, "w+") as text:
            data_iterator = unexpanded_data.iterrows()
            if interactive:
                data_iterator = tqdm(data_iterator, total=unexpanded_data.shape[0])
            for row in data_iterator:
                row_data = {trait: row[1][trait] for trait in traits_labels}
                for sentence in findall(r'(".+")|([^.?!]+[.?!])', row[1]['text']):
                    for match in sentence:
                        if match and len(match.split()) < max_length:
                            text.write(match.translate(translation_table).strip() + "\n")
                    data = data.append(row_data, ignore_index=True)
    else:
        data['text'].to_csv(text_file, index=False, header=False)
    numerical_traits = data[traits_labels].applymap(lambda x: 1 if x == 'y' else 0)
    for trait in traits_labels:
        numerical_traits[trait].to_csv(f'{label_file}_{trait[1:4]}',
                                       index=False, header=False)
    with open(vocab_file, "w+") as vocab:
        vocab.writelines(map(lambda x: x + '\n', tx.data.make_vocab(text_file)))


def prepare_mbti():
    ...


def prepare_personage_data():
    ...


def prepare_personality_detection():
    ...


def main():
    """Entrypoint.
    """
    DATASET2FUN = {'ear': prepare_ear,
                   'essays': prepare_essays,
                   'mbti': prepare_mbti,
                   'personage-data': prepare_personage_data,
                   'personality-detection': prepare_personality_detection,
                   'yelp': prepare_yelp}  # FIXME: remove yelp, only used for debugging
    DATASET2LINK = {'ear': ...,  # TODO
                    'essays': "https://github.com/yashsmehta/personality-prediction/blob/65b9d821b2c3f71e73fef77d4e9ef2117f990a8f/data/essays/essays.csv?raw=true",
                    'mbti': "https://github.com/yashsmehta/personality-prediction/blob/65b9d821b2c3f71e73fef77d4e9ef2117f990a8f/data/kaggle/kaggle.csv?raw=true",
                    'personage-data': "http://farm2.user.srcf.net/research/personage/personage-data.tar.gz",
                    'personality-detection': "https://raw.githubusercontent.com/emorynlp/personality-detection/3ec08a58dc7c708c5dfc314b3bff8f5808786928/CSV/friends-personality.csv",
                    'yelp': "https://drive.google.com/file/d/1HaUKEYDBEk6GlJGmXwqYteB-4rS9q8Lg/view?usp=sharing"}
    DOWNLOAD_IS_COMPRESSED = defaultdict(lambda: False,
                                         [('yelp', True), ('personage-data', True)])

    parser = argparse.ArgumentParser(description='Dataset downloader and preprocessor')
    parser.add_argument('--dataset',
                        default='yelp',
                        choices=list(DATASET2FUN.keys()),
                        help='name of the dataset to download or preprocess')
    parser.add_argument('--base-path',
                        help='base path for the dataset dir',
                        default='./personality_transfer_model/data')
    parser.add_argument('--max-length',
                        help='Max length (number of words) for a row in the dataset',
                        default=20)
    parser.add_argument('--interactive',
                        help='If true will use tqdm to show progress',
                        action='store_true')
    args = parser.parse_args()

    store_path = path.join(args.base_path, "original_datasets", args.dataset)
    if not path.exists(store_path):
        print("Creating destination folder for the dataset")
        makedirs(store_path)
    if not listdir(store_path):
        print("Starting download of dataset")
        tx.data.maybe_download(
            urls=DATASET2LINK[args.dataset],
            path=store_path,
            filenames=args.dataset,
            extract=DOWNLOAD_IS_COMPRESSED[args.dataset])
        # TODO: add remove of zip if dataset is compressed
    if not path.isdir(store_path):
        raise IOError("The path where the dataset should be store already exists and it's a file, not a folder.")
    store_processed_path = path.join(args.base_path, args.dataset)
    print("Starting preprocessing of dataset")
    if path.exists(store_processed_path) and path.isdir(store_processed_path):
        print("Dataset has already been processed, a probably unwanted behavior will follow...")
    else:
        mkdir(store_processed_path)
    DATASET2FUN[args.dataset](store_path,
                              args.max_length,
                              f"{store_processed_path}/text",
                              f"{store_processed_path}/labels",
                              f"{store_processed_path}/vocab",
                              args.interactive)
    print("Dataset preprocessed correctly")


if __name__ == '__main__':
    main()
