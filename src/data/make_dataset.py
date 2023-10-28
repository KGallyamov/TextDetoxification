import os
import urllib.request
import zipfile

import nltk
import pandas as pd
import torch
import torch.nn.functional as F
import torchtext
import youtokentome as yttm
from loguru import logger
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator
from tqdm.auto import tqdm
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AutoTokenizer, AutoModel

DATA_URL = 'https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip'
DATA_RAW_FOLDER = 'data/raw/'
DATA_INTERIM_FOLDER = 'data/interim/'
FILENAME = 'filtered.tsv'
PATH_TO_BPE_MODEL = 'models/bpe.model'
BPE_SEP = "_ "


def _download_dataset():
    """
    Download data from DATA_URL, unzip into DATA_RAW_FOLDER/FILENAME
    :return: None
    """
    zip_file, headers = urllib.request.urlretrieve(DATA_URL)

    if not os.path.exists(DATA_RAW_FOLDER):
        os.makedirs(DATA_RAW_FOLDER)

    # Extract the tsv file from the zip
    with zipfile.ZipFile(zip_file) as zf:
        zf.extract(FILENAME, path=DATA_RAW_FOLDER)

    logger.info('File downloaded and extracted to ' + str(os.path.join(DATA_RAW_FOLDER, FILENAME)))


def _prepare_dataset():
    """
    Split data from DATA_RAW_FOLDER/FILENAME into train-val-test, rename columns
    :return: None
    """
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
    """
    Train BPE on texts from dataframe and save it into PATH_TO_BPE_MODEL
    :param dataframe: Dataframe with texts on which BPE will be fitted
    :return: None
    """
    data_path = 'data/raw/sentences.txt'

    with open(data_path, "w") as out:
        for reference, translation in zip(dataframe.reference, dataframe.translation):
            print(f'{reference.lower()} {translation.lower()}', file=out)

    yttm.BPE.train(data_path,
                   PATH_TO_BPE_MODEL,
                   vocab_size=10000,
                   n_threads=-1)
    os.remove(data_path)  # remove file on which BPE was trained


class Evaluator:
    """
    Class to evaluate quality of detoxification
    """

    def __init__(self):
        """
        Load models and tokenizers for similarity and toxicity measurement
        """
        self.toxicity_tokenizer = RobertaTokenizer.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')
        self.toxicity_model = RobertaForSequenceClassification.from_pretrained(
            'SkolkovoInstitute/roberta_toxicity_classifier')
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.similarity_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        self.similarity_model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

    @staticmethod
    def bleu_score(src, tgt, bpe_sep: str = None):
        """
        :param src: List[src] / List[int] of source (reference) tokens / token ids
        :param tgt: List[src] / List[int] of target (translation) tokens / token ids
        :param bpe_sep: If dataset uses BPE, concat sub-words using this separator
        :return: float, BLEU score for (src, tgt)
        """
        smoothing_functions = SmoothingFunction()
        if bpe_sep is not None:
            assert isinstance(src[0], str) and isinstance(tgt[0], str), \
                'If computing BLEU for BPE-tokenized sentence, pass List[str], not indices'
            assert bpe_sep[-1] == ' ', 'Append space to BPE separator'
            src = (' '.join(src)).replace(bpe_sep, '').split()
            tgt = (' '.join(tgt)).replace(bpe_sep, '').split()

        bleu = sentence_bleu([src], tgt, smoothing_function=smoothing_functions.method1)
        return bleu

    def estimate_toxicity(self, sentences_batch, threshold: float = 0.5, probs: bool = False):
        """
        Code adapted from https://github.com/s-nlp/detox/blob/main/emnlp2021/metric/metric.py
        :param probs: bool, If set to True, the return value will be softmax(logits). Else return {0, 1} predicions
        :param threshold: Cutoff for probabilities of toxicity. Has no effect is probs=True
        :param sentences_batch: List[str] of sentences
        :return: Array of length sentences_batch
        """
        batch = self.toxicity_tokenizer(sentences_batch, return_tensors='pt', padding=True).to(self.device)
        with torch.inference_mode():
            logits = self.toxicity_model(**batch).logits

        if probs:
            scores = torch.softmax(logits, -1)[:, 1].cpu().numpy()
        else:
            scores = (torch.softmax(logits, -1)[:, 1] > threshold).cpu().numpy()
        return scores

    def estimate_similarity(self, source_sentences_batch, predicted_sentences_batch):
        """
        Code adapted from https://huggingface.co/sentence-transformers/all-mpnet-base-v2
        :param source_sentences_batch: List[str] of reference (source) sentences
        :param predicted_sentences_batch: List[str] of translation (target) sentences
        :return: Array of same length as source_sentences_batch and predicted_sentences_batch
        """
        encoded_source = self.similarity_tokenizer(source_sentences_batch, padding=True, truncation=True,
                                                   return_tensors='pt').to(self.device)
        encoded_pred = self.similarity_tokenizer(predicted_sentences_batch, padding=True, truncation=True,
                                                 return_tensors='pt').to(self.device)

        def _mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1),
                                                                                      min=1e-9)

        with torch.no_grad():
            source_emb = self.similarity_model(**encoded_source)
            pred_emb = self.similarity_model(**encoded_pred)

        source_emb = _mean_pooling(source_emb, encoded_source['attention_mask'])
        source_emb = F.normalize(source_emb, p=2, dim=1)

        pred_emb = _mean_pooling(pred_emb, encoded_pred['attention_mask'])
        pred_emb = F.normalize(pred_emb, p=2, dim=1)

        similarities = (source_emb * pred_emb).sum(-1)
        return similarities


class TextDetoxificationDataset(Dataset):
    def __init__(self,
                 mode: str = 'train',
                 download: bool = False,
                 low_difference_filter: float = 0.0,
                 use_bpe: bool = False,
                 vocab: torchtext.vocab.Vocab = None,
                 char_level: bool = False
                 ):
        """
        :param mode: One of 'train', 'val' and 'test'. If set to train, vocab and tokenizer is fit on the data
        :param download: If set to True, download from DATA_URL. Else, look in DATA_RAW_FOLDER and DATA_INTERIM_FOLDER
        :param low_difference_filter: Cutoff for (src, tgt) toxicity drop.
            If greater than zero, pairs with |src_tox - ref_tox| <= low_difference_filter will be discarded
        :param use_bpe: If set to True, fit BPE from YouTokenToMe on the data. Else, use nltk.word_tokenize
        :param vocab: If mode is not 'train', in this argument should be the vocab of the training dataset
        :param char_level: If set to True, build vocab of characters
        """
        assert mode == 'train' or vocab is not None, 'For non-training mode, pass the Vocab from train dataset'
        assert not (use_bpe and os.path.exists(
            PATH_TO_BPE_MODEL)) or mode == 'train', 'BPE tokenizer should be fitted on train'
        assert 0.0 <= low_difference_filter < 1.0, 'Toxicity difference threshold should be from [0; 1)'
        assert not use_bpe if char_level else True, 'Incompatible tokenization types'

        nltk.download('punkt', quiet=True)
        if not os.path.exists(DATA_RAW_FOLDER + FILENAME) or download:
            logger.info(f'Downloading the data from {DATA_URL}')
            _download_dataset()
        if not os.path.exists(DATA_INTERIM_FOLDER + f'{mode}.tsv') or download:
            logger.info(f'Splitting into train-val-test')
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
            min_count = 10 if not self.use_bpe else 1  # Cutoff rare words

            def _iterator():
                for reference, reference in tqdm(zip(self.data.reference, self.data.translation),
                                                 desc='Collecting vocab'):
                    yield self._tokenize_sentence(reference) + self._tokenize_sentence(reference)

            logger.info('Started building vocab')
            if not self.use_bpe:
                vocab = build_vocab_from_iterator(_iterator(), min_freq=min_count, specials=self.specials)
            else:
                vocab = build_vocab_from_iterator(iter(self.bpe_model.vocab()), min_freq=min_count,
                                                  specials=self.specials)
            vocab.set_default_index(self.UNK_IDX)
            logger.info('Vocab built successfully')

        self.vocab = vocab

    def _tokenize_sentence(self, sentence):
        """
        Apply BPE or nltk.word_tokenize on the sentence.lower()
        :param sentence: str
        :return: Tokenized sentence
        """
        # casing is unlikely to be significant for the task
        if self.use_bpe:
            tokens = self.bpe_model.encode([sentence.lower()], output_type=yttm.OutputType.SUBWORD)
        elif self.char_level:
            tokens = list(sentence.lower())
        else:
            tokens = nltk.word_tokenize(sentence.lower())
        return tokens

    def detokenize(self, tokens_batch):
        """
        Transform predicted token ids to List[str] of sentences omitting special tokens
        :param tokens_batch: Token ids of shape [batch_size, seq_len]
        :return: List[str]
        """
        itos = self.vocab.get_itos()
        sentences = []
        for sequence in tokens_batch:
            sentence = []
            for word_id in sequence:
                if word_id in [self.BOS_IDX, self.PAD_IDX]:
                    continue
                if word_id == self.EOS_IDX:
                    break
                sentence.append(itos[word_id])
            if len(sentence) == 0:
                sentence = ['<pad>']
            sentences.append(" ".join(sentence))
        if self.use_bpe:
            sentences = [sentence.replace(BPE_SEP, '') for sentence in sentences]
        return sentences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        # Select high toxicity sentences
        source, target = row['reference'], row['translation']
        if row['ref_tox'] < row['trn_tox']:
            source, target = row['translation'], row['reference']

        # Tokenize sentences and transform to indices
        source_tokens = self._tokenize_sentence(source)
        target_tokens = self._tokenize_sentence(target)

        source_indices = self.vocab(source_tokens)
        target_indices = self.vocab(target_tokens)

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
