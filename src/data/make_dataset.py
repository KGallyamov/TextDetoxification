import os
import urllib.request
import zipfile

import nltk
import pandas as pd
import torch
import torchtext
import youtokentome as yttm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator
from tqdm.auto import tqdm

DATA_URL = 'https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip'
DATA_RAW_FOLDER = 'data/raw/'
DATA_INTERIM_FOLDER = 'data/interim/'
FILENAME = 'filtered.tsv'
PATH_TO_BPE_MODEL = 'models/bpe.model'


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

    if not os.path.exists(DATA_INTERIM_FOLDER):
        os.makedirs(DATA_INTERIM_FOLDER)

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


def bleu_score(src, tgt, bpe_sep: str = None):
    """

    :param src:
    :param tgt:
    :param bpe_sep:
    :return:
    """
    smoothing_functions = SmoothingFunction()
    if bpe_sep is not None:
        assert isinstance(src[0], src) and isinstance(tgt[0], src),\
            'If computing BLEU for BPE-tokenized sentence, pass List[str], not indices'
        assert bpe_sep[-1] == ' ', 'Append space to BPE separator'
        src = (' '.join(src)).replace(bpe_sep, '').split()
        tgt = (' '.join(tgt)).replace(bpe_sep, '').split()

    bleu = sentence_bleu([src], tgt, smoothing_function=smoothing_functions.method1)
    return bleu


def estimate_toxicity(sentences_batch):
    """

    :param sentences_batch:
    :return:
    """
    pass


def estimate_similarity(source_sentences_batch, target_sentences_batch):
    """

    :param source_sentences_batch:
    :param target_sentences_batch:
    :return:
    """
    pass


class TextDetoxificationDataset(Dataset):
    def __init__(self,
                 mode: str = 'train',
                 download: bool = False,
                 low_difference_filter: float = 0.0,
                 use_bpe: bool = False,
                 vocab: torchtext.vocab.Vocab = None,
                 char_level: bool = False
                 ):
        assert mode == 'train' or vocab is not None, 'For non-training mode, pass the Vocab from train dataset'
        assert not (use_bpe and os.path.exists(
            PATH_TO_BPE_MODEL)) or mode == 'train', 'BPE tokenizer should be fitted on train'
        assert 0.0 <= low_difference_filter < 1.0, 'Toxicity difference threshold should be from [0; 1)'
        assert not use_bpe if char_level else True, 'Incompatible tokenization types'

        nltk.download('punkt')
        if not os.path.exists(DATA_RAW_FOLDER + FILENAME) or download:
            print(f'Downloading the data from {DATA_URL}')
            _download_dataset()
        if not os.path.exists(DATA_INTERIM_FOLDER + f'{mode}.tsv') or download:
            print(f'Splitting into train-val-test')
            _prepare_dataset()

        self.mode = mode
        self.use_bpe = use_bpe
        self.char_level = char_level

        self.data = pd.read_csv(f'{DATA_INTERIM_FOLDER}/{mode}.tsv', sep='\t')

        if low_difference_filter != 0:
            self.data = self.data[abs(self.data.ref_tox - self.data.trn_tox) > low_difference_filter]
        assert len(self.data), f'Change low_difference_filter, {low_difference_filter} is too high'

        if self.use_bpe and not os.path.exists(PATH_TO_BPE_MODEL):
            _prepare_bpe_tokenizer(self.data)
            self.bpe_model = yttm.BPE(PATH_TO_BPE_MODEL, n_threads=2)

        self.PAD_IDX, self.UNK_IDX, self.BOS_IDX, self.EOS_IDX = 0, 1, 2, 3
        self.specials = ['<pad>', '<unk>', '<bos>', '<eos>']

        if vocab is None:
            min_count = 10 if not self.use_bpe else 1

            def _iterator():
                for i, row in tqdm(self.data.iterrows(), desc='Collecting vocab'):
                    yield self._tokenize_sentence(row['reference']) + self._tokenize_sentence(row['translation'])

            print('Started building vocab')
            if not self.use_bpe:
                vocab = build_vocab_from_iterator(_iterator(), min_freq=min_count, specials=self.specials)
            else:
                vocab = build_vocab_from_iterator(iter(self.bpe_model.vocab()), min_freq=min_count,
                                                  specials=self.specials)
            vocab.set_default_index(self.UNK_IDX)
            print('Vocab built successfully')

        self.vocab = vocab

    def _tokenize_sentence(self, sentence):
        # casing is unlikely to be significant for the task
        if self.use_bpe:
            tokens = self.bpe_model.encode([sentence.lower()], output_type=yttm.OutputType.SUBWORD)
        elif self.char_level:
            tokens = list(sentence.lower())
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

        source_indices = [self.BOS_IDX] + self.vocab(source_tokens) + [self.EOS_IDX]
        target_indices = [self.BOS_IDX] + self.vocab(target_tokens) + [self.EOS_IDX]

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
    train_set = TextDetoxificationDataset(download=False)
    for t in train_set:
        pass
    for t in train_set:
        print(t)
        break
