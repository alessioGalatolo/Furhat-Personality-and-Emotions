import argparse
from collections import defaultdict
from os import makedirs, mkdir, path,  listdir, remove
from re import findall, sub
import string
import pandas as pd
from tqdm import tqdm
import texar.torch as tx

translation_table = defaultdict(lambda: ' ', {ord(letter): letter for letter in string.ascii_letters})
translation_table[ord('.')] = ' . '
translation_table[ord('?')] = ' ? '
translation_table[ord(',')] = ' , '
translation_table[ord('!')] = ' ! '


def parse_sentence(text, max_length=None):
    # remove links
    text = sub(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)',
               '', text)
    # split sentences
    for sentence in findall(r'(".+")|([^.?!]+[.?!])', text):
        for match in sentence:
            if match:
                translated = sub(r' +', ' ', match.translate(translation_table).strip())
                if len(translated) > 2:
                    if max_length is None or len(translated.split()) < max_length:
                        yield translated


def prepare_essays(base_path, max_length, text_file, label_file, interactive):
    traits_labels = ['cOPN', 'cCON', 'cEXT', 'cAGR', 'cNEU']
    with open(f"{base_path}/essays", "r") as essays:
        data = pd.read_csv(essays)
    print('Expanding the essays into single sentences, this will probably take a long time...')
    unexpanded_data = data

    data = pd.DataFrame()
    with open(text_file, "w+") as text:
        data_iterator = unexpanded_data.iterrows()
        if interactive:
            data_iterator = tqdm(data_iterator, total=unexpanded_data.shape[0])
        for row in data_iterator:
            row_data = {trait: row[1][trait] for trait in traits_labels}
            for sentence in parse_sentence(row[1]['text'], max_length):
                text.write(sentence + "\n")
                data = data.append(row_data, ignore_index=True)
    numerical_traits = data[traits_labels].applymap(lambda x: 1 if x == 'y' else 0)
    for trait in traits_labels:
        numerical_traits[trait].to_csv(f'{label_file}_{trait[1:4]}',
                                       index=False, header=False)


def prepare_mbti(base_path, max_length, text_file, label_file, interactive):
    type2trait = {'I': {'EXT': 0}, 'E': {'EXT': 1}}
    with open(f"{base_path}/mbti", "r") as mbti:
        data = pd.read_csv(mbti)
    print('Expanding mbti into single sentences, this will probably take a long time...')
    unexpanded_data = data

    data = pd.DataFrame()
    with open(text_file, "w+") as text:
        data_iterator = unexpanded_data.iterrows()
        if interactive:
            data_iterator = tqdm(data_iterator, total=unexpanded_data.shape[0])
        for row in data_iterator:
            row_data = type2trait[row[1]['type'][0]]
            for post in row[1]['text'].split('|||'):
                for sentence in parse_sentence(post, max_length):
                    text.write(sentence + "\n")
                    data = data.append(row_data, ignore_index=True)
    data['EXT'].astype(int).to_csv(f'{label_file}_EXT',
                                   index=False, header=False)


def _prepare_personage_data(path, max_length, interactive):
    with open(path, 'r') as personage:
        data = pd.read_csv(personage, delimiter='\t')
    texts = []
    labels = []
    data_iterator = data.iterrows()
    if interactive:
        data_iterator = tqdm(data_iterator, total=data.shape[0])
    for row in data_iterator:
        row_data = "1\n" if row[1]['avg.extra'] > 4 else "0\n"
        for sentence in parse_sentence(row[1]['realization'], max_length):
            texts.append(sentence+'\n')
            labels.append(row_data)
    return texts, labels


def prepare_personage_data(base_path, max_length, text_file, label_file, interactive):
    texts, labels = _prepare_personage_data(f"{base_path}/predefinedParams.tab",
                                            max_length, interactive)
    texts2, labels2 = _prepare_personage_data(f"{base_path}/randomParams.tab",
                                              max_length, interactive)
    texts.extend(texts2)
    labels.extend(labels2)
    with open(f'{label_file}_EXT', 'w+') as label:
        label.writelines(labels)
    with open(text_file, 'w+') as text:
        text.writelines(texts)


def prepare_personality_detection(base_path, max_length, text_file, label_file, interactive):
    with open(f"{base_path}/personality-detection", "r") as data_file:
        data = pd.read_csv(data_file)
    unexpanded_data = data

    data = pd.DataFrame()
    with open(text_file, "w+") as text:
        data_iterator = unexpanded_data.iterrows()
        if interactive:
            data_iterator = tqdm(data_iterator, total=unexpanded_data.shape[0])
        for row in data_iterator:
            row_data = {'EXT': row[1]['cEXT']}
            character = row[1]['character']
            next_is_text = False
            texts = []
            for row_part in row[1]['text'].split('b>'):
                if next_is_text:
                    # remove initial : and ending <br><br>
                    row_part = row_part[2:-9]
                    # remove text in parenthesis indicating what's happening in the scene
                    row_part = sub(r'\(.+\)', '', row_part)
                    texts.append(row_part)
                    next_is_text = False
                elif str(row_part).startswith(character):
                    next_is_text = True
            for post in texts:
                for sentence in parse_sentence(post, max_length):
                    text.write(sentence + "\n")
                    data = data.append(row_data, ignore_index=True)
    data['EXT'].astype(int).to_csv(f'{label_file}_EXT',
                                   index=False, header=False)


def main():
    """Entrypoint.
    """
    DATASET2FUN = {'essays': prepare_essays,
                   'mbti': prepare_mbti,
                   'personage-data': prepare_personage_data,
                   'personality-detection': prepare_personality_detection}
    DATASET2LINK = {'essays': "https://github.com/yashsmehta/personality-prediction/blob/65b9d821b2c3f71e73fef77d4e9ef2117f990a8f/data/essays/essays.csv?raw=true",
                    'mbti': "https://github.com/yashsmehta/personality-prediction/blob/65b9d821b2c3f71e73fef77d4e9ef2117f990a8f/data/kaggle/kaggle.csv?raw=true",
                    'personage-data': "http://farm2.user.srcf.net/research/personage/personage-data.tar.gz",
                    'personality-detection': "https://raw.githubusercontent.com/emorynlp/personality-detection/3ec08a58dc7c708c5dfc314b3bff8f5808786928/CSV/friends-personality.csv"}
    DOWNLOAD_IS_COMPRESSED = defaultdict(lambda: False,
                                         [('personage-data', True)])

    parser = argparse.ArgumentParser(description='Dataset downloader and preprocessor')
    parser.add_argument('--dataset',
                        choices=list(DATASET2FUN.keys()),
                        help='name of the dataset to download or preprocess')
    parser.add_argument('--base-path',
                        help='base path for the dataset dir',
                        default='./style_transfer_paraphrase/utils/data')
    parser.add_argument('--max-length',
                        help='Max length (number of words) for a row in the dataset. Use 50 if data for paraphrase model.',
                        default=20)
    parser.add_argument('--interactive',
                        help='If true will use tqdm to show progress',
                        action='store_true')
    parser.add_argument('--train-test-split',
                        help='If true will split the final dataset into train, dev, test. Use if data for paraphrase model.',
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
        if DOWNLOAD_IS_COMPRESSED[args.dataset]:
            remove(path.join(store_path, args.dataset))
    if not path.isdir(store_path):
        raise IOError("The path where the dataset should be store already exists and it's a file, not a folder.")
    store_processed_path = path.join(args.base_path, args.dataset)
    print("Starting preprocessing of dataset")
    if path.exists(store_processed_path) and path.isdir(store_processed_path):
        print("Dataset has already been processed, a probably unwanted behavior will follow...")
    else:
        mkdir(store_processed_path)
    text_file = f"{store_processed_path}/text"
    label_file = f"{store_processed_path}/labels"
    DATASET2FUN[args.dataset](store_path,
                              int(args.max_length),
                              text_file,
                              label_file,
                              args.interactive)

    with open(f"{store_processed_path}/vocab", "w+") as vocab:
        vocab.writelines(map(lambda x: x + '\n', tx.data.make_vocab(text_file)))
    if args.train_test_split:
        print("Doing trianing testing split")
        # FIXME: this is slow and can probably be improved
        data = pd.read_csv(text_file, sep='<', names=['text'], header=None)
        data['label'] = pd.read_csv(f'{label_file}_EXT', sep='<', header=None)
        train = data.sample(frac=0.8)
        val_test = data.drop(train.index)
        val = val_test.sample(frac=0.5)
        test = val_test.drop(val.index)

        train['text'].to_csv(f"{store_processed_path}/train.txt",
                             index=False, header=False, sep='<')
        val['text'].to_csv(f"{store_processed_path}/dev.txt",
                           index=False, header=False, sep='<')
        test['text'].to_csv(f"{store_processed_path}/test.txt",
                            index=False, header=False, sep='<')
        train['label'].to_csv(f"{store_processed_path}/train.label",
                              index=False, header=False)
        val['label'].to_csv(f"{store_processed_path}/dev.label",
                            index=False, header=False)
        test['label'].to_csv(f"{store_processed_path}/test.label",
                             index=False, header=False)
    print("Dataset preprocessed correctly")


if __name__ == '__main__':
    main()
