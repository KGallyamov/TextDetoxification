import os
import urllib.request
import zipfile
from collections import Counter

import nltk
import pandas as pd
import torch
import youtokentome as yttm
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

DATA_URL = 'https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip'
DATA_RAW_FOLDER = 'data/raw/'
DATA_INTERIM_FOLDER = 'data/interim/'
FILENAME = 'filtered.tsv'
PATH_TO_BPE_MODEL = 'models/bpe.model'
PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3


def _download_dataset():
    zip_file, headers = urllib.request.urlretrieve(DATA_URL)

    if not os.path.exists(DATA_RAW_FOLDER):
        os.makedirs(DATA_RAW_FOLDER)

    # Extract the tsv file from the zip
    with zipfile.ZipFile(zip_file) as zf:
        zf.extract(FILENAME, path=DATA_RAW_FOLDER)

    print('File downloaded and extracted to', os.path.join(DATA_RAW_FOLDER, FILENAME))


def _prepare_dataset():
    # room for improvement: apply spellchecking (like SAGE) to sentences
    data = pd.read_csv(DATA_RAW_FOLDER + FILENAME, sep='\t')
    del data["Unnamed: 0"]
    data["length_diff"] = data["lenght_diff"]
    del data["lenght_diff"]
    data.to_csv(DATA_INTERIM_FOLDER + FILENAME, sep='\t')
    train, val_test = train_test_split(data, train_size=0.8, random_state=42)
    val, test = train_test_split(val_test, test_size=0.5, random_state=42)

    train.to_csv(DATA_INTERIM_FOLDER + 'train.tsv', sep='\t')
    val.to_csv(DATA_INTERIM_FOLDER + 'val.tsv', sep='\t')
    test.to_csv(DATA_INTERIM_FOLDER + 'test.tsv', sep='\t')


def _prepare_bpe_tokenizer(dataframe):
    data_path = 'data/raw/sentences.txt'

    with open(data_path, "w") as out:
        for i in range(len(dataframe)):
            row = dataframe.iloc[i]
            print(f'{row.reference.lower()} {row.translation.lower()}', file=out)

    yttm.BPE.train(data_path,
                   PATH_TO_BPE_MODEL,
                   vocab_size=10000,
                   n_threads=4)
    os.remove(data_path)


class TextDetoxificationDataset(Dataset):
    def __init__(self,
                 mode: str = 'train',
                 download: bool = False,
                 low_difference_filter: float = 0.0,
                 use_bpe: bool = False,
                 token2idx: dict = None
                 ):
        assert mode == 'train' or token2idx is not None, 'For non-training mode, pass token2idx from train dataset'
        assert use_bpe and os.path.exists(PATH_TO_BPE_MODEL) or mode == 'train', 'BPE tokenizer is fitted on train'
        assert 0.0 <= low_difference_filter < 1.0, 'Toxicity difference threshold should be from [0; 1)'

        nltk.download('punkt')
        if not os.path.exists(DATA_RAW_FOLDER + FILENAME) or download:
            print(f'Downloading the data from {DATA_URL}')
            _download_dataset()
        if not os.path.exists(DATA_INTERIM_FOLDER + f'{mode}.tsv') or download:
            print(f'Splitting into train-val-test')
            _prepare_dataset()

        self.mode = mode
        self.use_bpe = use_bpe

        self.data = pd.read_csv(f'{DATA_INTERIM_FOLDER}/{mode}.tsv', sep='\t')

        if low_difference_filter != 0:
            self.data = self.data[abs(self.data.ref_tox - self.data.trn_tox) > low_difference_filter]
        assert len(self.data), f'Change low_difference_filter, {low_difference_filter} is too high'

        if self.use_bpe and not os.path.exists(PATH_TO_BPE_MODEL):
            _prepare_bpe_tokenizer(self.data)
            self.bpe_model = yttm.BPE(PATH_TO_BPE_MODEL, n_threads=2)

        if token2idx is None:
            # Add torchtext.vocab import build_vocab_from_iterator if loader is the bottleneck
            tokens = ['<pad>', '<unk>', '<bos>', '<eos>']
            tokens_freq = Counter()

            for i in tqdm(range(len(self.data)), desc='Building vocab'):
                row = self.data.iloc[i]
                tokens_freq.update(self._tokenize_sentence(row['reference']))
                tokens_freq.update(self._tokenize_sentence(row['translation']))

            if not use_bpe:
                min_count = 10
                tokens.extend(list(sorted(t for t, c in tokens_freq.items() if c >= min_count)))
            else:
                tokens.extend([t for t, c in tokens_freq.items()])

            token2idx = {token: i for i, token in enumerate(tokens)}
        self.token2idx = token2idx
        self.idx2token = {i: token for token, i in self.token2idx.items()}

    def _tokenize_sentence(self, sentence):
        # casing is unlikely to be significant for the task
        if self.use_bpe:
            tokens = self.bpe_model.encode([sentence.lower()], output_type=yttm.OutputType.SUBWORD)
        else:
            tokens = nltk.word_tokenize(sentence.lower())
        return tokens

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        source, target = row['reference'], row['translation']
        if row['ref_tox'] < row['trn_tox']:
            source, target = row['translation'], row['reference']

        source_tokens = self._tokenize_sentence(source)
        target_tokens = self._tokenize_sentence(target)

        source_indices = [BOS_IDX] + [self.token2idx.get(token, '<unk>') for token in source_tokens] + [EOS_IDX]
        target_indices = [BOS_IDX] + [self.token2idx.get(token, '<unk>') for token in target_tokens] + [EOS_IDX]

        stats = [
            row['similarity'],
            min(row['ref_tox'], row['trn_tox']),
            max(row['ref_tox'], row['trn_tox'])
        ]

        source = torch.Tensor(source_indices).long()
        target = torch.Tensor(target_indices).long()
        stats = torch.Tensor(stats)

        return source, target, stats


if __name__ == '__main__':
    # use case
    train_set = TextDetoxificationDataset(download=True)
    print(train_set[0])
    print(train_set.idx2token[4])
