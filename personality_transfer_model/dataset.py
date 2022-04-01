import itertools
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import texar.torch as tx


class TextDataset(Dataset):
    def __init__(self, text_filename, label_filename, vocab_filename,
                 seed=None, balance_data=True, balancing_method='undersampling'):
        self.vocab = TextDataset.load_vocab(vocab_filename)
        with open(text_filename, mode='r') as text_file,\
             open(label_filename, 'r') as label_file:
            texts = (f'{self.vocab.bos_token} {line[:-1]} {self.vocab.eos_token}'.split() for line in text_file)
            labels = (f'{line[:-1]}' for line in label_file)
            self.text_labels = pd.DataFrame({'text': texts, 'label': labels})
        self.text_labels['label'] = pd.to_numeric(self.text_labels['label'])
        self.length = len(self.text_labels)
        if balance_data:
            if balancing_method == 'undersampling':
                positive_examples = self.text_labels[self.text_labels['label'] == 1]
                negative_examples = self.text_labels[self.text_labels['label'] == 0]
                self.length = min(len(positive_examples), len(negative_examples))
                positive_examples = positive_examples.sample(frac=1, random_state=seed).reset_index(drop=True)
                negative_examples = negative_examples.sample(frac=1, random_state=seed).reset_index(drop=True)
                positive_examples = positive_examples[:self.length]
                negative_examples = negative_examples[:self.length]
                self.text_labels = positive_examples.append(negative_examples, ignore_index=True)
            else:
                raise NotImplementedError

        input_lens = []
        for row in self.text_labels.iterrows():
            input_lens.append(len(row[1]['text']))
        self.text_labels['length'] = input_lens
        self.input_len = self.text_labels['length'].max() - 1

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        text = self.text_labels[index: index+1]['text'][index].copy()
        length = self.text_labels[index: index+1]['length'][index]
        label = self.text_labels[index: index+1]['label'][index]
        text_ids = TextDataset.text2ids(text, self.vocab)
        return {'text': text, 'text_ids': text_ids,
                'label': label, 'length': length}

    @staticmethod
    def text2ids(text_tokens, vocab, input_len=0):
        if text_tokens[0] != vocab.bos_token:
            text_tokens.insert(0, vocab.bos_token)
        if text_tokens[-1] != vocab.eos_token and text_tokens[-1] != '':
            text_tokens.append(vocab.eos_token)
        if len(text_tokens) < input_len+1:
            for _ in range(input_len-len(text_tokens)+1):
                text_tokens.append('')
        text_ids = vocab.map_tokens_to_ids_py(text_tokens)
        return text_ids

    @staticmethod
    def load_vocab(vocab_file):
        vocab_hp = tx.HParams({'vocab_file': vocab_file},
                              tx.data.data.multi_aligned_data._default_dataset_hparams())
        return tx.data.MultiAlignedData.make_vocab([vocab_hp])[0]


class TextDataLoader():
    def __init__(self, dataset, batch_size, device):
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size

    def __iter__(self):
        self.n = 0
        self.idx_perm = torch.randperm(len(self.dataset))
        return self

    def __next__(self):
        if self.n > len(self.dataset):
            raise StopIteration

        text = []
        text_ids = []
        labels = []
        length = []
        for i in range(self.n, self.n+self.batch_size):
            row = self.dataset[i]
            text.append(row['text'])
            text_ids.append(row['text_ids'])
            labels.append(row['label'])
            length.append(row['length'])
        text_ids = np.array(list(itertools.zip_longest(*text_ids,
                                                       fillvalue=self.dataset.vocab.unk_token_id))).T
        result = {'text': list(map(list,
                                   zip(*list(itertools.zip_longest(*text,
                                                                   fillvalue=''))))),
                  'text_ids': torch.LongTensor(text_ids).to(self.device),
                  'labels': torch.LongTensor(labels).to(self.device),
                  'length': torch.LongTensor(length).to(self.device)}
        self.n += self.batch_size
        return result
