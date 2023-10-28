import datetime
import os
import random
import sys

import gensim.downloader as api
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict
from fire import Fire
from loguru import logger
from lovely_numpy import lo
from peft import get_peft_model, TaskType, LoraConfig
from torch import nn
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq

GLOVE_WIKI_EMBEDDING_DIM = 300
GLOVE_TWITTER_EMBEDDING_DIM = 200
T5_MAX_LENGTH = 128
T5_CHECKPOINT = "t5-small"


class BaselineTranslationModel(nn.Module):
    # This code is adapted from:
    # https://github.com/yandexdataschool/nlp_course/blob/2022/week04_seq2seq/practice_and_homework_pytorch.ipynb

    def __init__(self, vocab, emb_dim, hidden_dim, n_layers, do_weight_tying=True):
        nn.Module.__init__(self)
        self.num_words = len(vocab)
        self.hidden_dim = hidden_dim
        self.vocab = vocab
        self.emb_dim = emb_dim
        self.embeddings = nn.Embedding(self.num_words, emb_dim)
        # Pretrained embeddings to speed up convergence
        if emb_dim in [GLOVE_TWITTER_EMBEDDING_DIM, GLOVE_WIKI_EMBEDDING_DIM]:
            self._init_embeddings()

        self.encoder = nn.GRU(emb_dim, hidden_dim, n_layers, batch_first=True)
        self.enc_dec_proj = nn.Linear(hidden_dim, hidden_dim)

        self.decoder = nn.GRUCell(hidden_dim, hidden_dim)
        self.vocab_projection = nn.Linear(emb_dim, self.num_words)

        # weight tying is done to reduce memory consumption
        if do_weight_tying:
            self.vocab_projection.weight = self.embeddings.weight

    def _init_embeddings(self):
        if self.emb_dim == GLOVE_WIKI_EMBEDDING_DIM:
            glove_model = api.load('glove-wiki-gigaword-300')
        else:
            glove_model = api.load('glove-twitter-200')
        for word in tqdm(self.vocab.get_itos(), desc="Initializing with pretrained embeddings"):
            if word in glove_model:
                self.embeddings.weight.data[self.vocab[word]] = torch.Tensor(glove_model.get_vector(word))
        assert self.embeddings.weight.requires_grad, "Embeddings should be trainable"

    def forward(self, inp, out):
        initial_state = self.encode(inp)
        return self.decode(initial_state, out)

    def encode(self, inp):
        emb = self.embeddings(inp)

        encoded, _ = self.encoder(emb)

        # take the first before padding
        lengths = (inp != self.vocab['<eos>']).to(torch.int64).sum(dim=1).clamp_max(inp.shape[1] - 1)
        last_state = encoded[torch.arange(len(encoded)), lengths]

        dec_start = self.enc_dec_proj(last_state)
        return [dec_start]

    def decode_step(self, prev_state, prev_output):
        inp_emb = self.embeddings(prev_output)
        new_dec_state = self.decoder(inp_emb, prev_state[0])
        output_logits = self.vocab_projection(new_dec_state)

        return [new_dec_state], output_logits

    def decode(self, initial_state, out_tokens):
        batch_size = out_tokens.shape[0]
        state = initial_state

        # initial logits: always predict BOS
        onehot_bos = F.one_hot(torch.full([batch_size], self.vocab['<bos>'], dtype=torch.int64),
                               num_classes=self.num_words).to(device=out_tokens.device)
        first_logits = torch.log(onehot_bos.to(torch.float32) + 1e-9)

        logits_sequence = [first_logits]
        for i in range(out_tokens.shape[1] - 1):
            state, logits = self.decode_step(state, out_tokens[:, i])  # teacher forcing
            logits_sequence.append(logits)
        return torch.stack(logits_sequence, dim=1)

    def decode_inference(self, initial_state, max_len=100, top_p=0.8):
        batch_size, device = len(initial_state[0]), initial_state[0].device
        state = initial_state
        outputs = [torch.full([batch_size], self.vocab['<bos>'], dtype=torch.long,
                              device=device)]
        all_states = [initial_state]
        for i in range(max_len):
            state, logits = self.decode_step(state, outputs[-1])
            probs = F.softmax(logits, -1)
            if top_p == -1:
                outputs.append(logits.argmax(-1))
                all_states.append(state)
                continue

            output = torch.zeros((batch_size,), dtype=torch.int64, device=device)
            for b in range(batch_size):  # nucleus sampling
                token_probs = probs[b]
                s_probs, indices = torch.sort(token_probs, descending=True, dim=0)
                idx = (torch.cumsum(s_probs, dim=0) > top_p).nonzero().squeeze()[0]
                token_idx = torch.multinomial(s_probs[:idx + 1] / s_probs[:idx + 1].sum(), num_samples=1)
                output[b] = indices[token_idx]
            outputs.append(output)
            all_states.append(state)

        return torch.stack(outputs, dim=1), all_states


def _inference_model(model, batch, test_dataset, tokenizer=None):
    """
    Get model predictions without teacher forcing
    :param model: encoder-recoder NN
    :param batch: data for the model
    :param test_dataset:
    :return: Tuple[List[str], List[str]] source and prediction sentences lists
    """
    if isinstance(model, BaselineTranslationModel):
        src, tgt = batch
        encoded = model.encode(src)
        pred, _ = model.decode_inference(encoded)
        pred_tokens = test_dataset.detokenize(pred)
        src_tokens = test_dataset.detokenize(src)
    else:
        outputs = model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], do_sample=True,
                                 top_p=0.8, max_length=100)
        pred_tokens = tokenizer.batch_decode(outputs.cpu().numpy(), skip_special_tokens=True)
        src_tokens = tokenizer.batch_decode(batch['input_ids'].cpu().numpy(), skip_special_tokens=True)
    return src_tokens, pred_tokens


def _get_batch_loss(model, batch, return_pred=False):
    """
    Forward batch through the model in teacher-forcing and compute loss
    :param model: encoder-decoder NN
    :param batch: input and labels
    :param return_pred: If set to True, th function will return model's predicted sequence and ground truth token ids
    :return: loss and (optionally) predicted and target token ids
    """
    if isinstance(model, BaselineTranslationModel):
        src, tgt = batch
        pred = model(src, tgt)
        loss = _compute_loss(pred, tgt, model.vocab['<pad>'])
        pred = pred.argmax(-1)
    else:
        model_outputs = model(**batch)
        loss = model_outputs.loss
        pred = model_outputs.logits.argmax(-1)
        tgt = batch['labels']

    if return_pred:
        return loss, pred, tgt
    return loss


def _compute_loss(logits_seq, out, pad_idx):
    """
    Calculate Cross-Entropy loss for seq2seq
    :param logits_seq: tokens ids of shape [batch_size, out_len, vocab_size]
    :param out: ground trutth of shape [batch_size, out_len]
    :param pad_idx: Tokens with this index will be masked out (ignored) from loss
    :return: float, CrossEntropy(logits, out)
    """
    mask = out != pad_idx
    vocab_size = logits_seq.shape[-1]
    targets_1hot = F.one_hot(out, vocab_size).to(torch.float32)

    # log-probabilities of all tokens at all steps, [batch_size, out_len, num_tokens]
    logprobs_seq = torch.log(logits_seq.softmax(dim=-1))

    # log-probabilities of correct outputs, [batch_size, out_len]
    logp_out = (logprobs_seq * targets_1hot).sum(dim=-1)

    # average cross-entropy over non-padding tokens
    return - torch.masked_select(logp_out, mask).mean()


def train_step(train_loader, model, metrics, optimizer, scheduler, gradient_accumulation_steps, device):
    """
    Iterate one training epoch
    :param train_loader: DataLoader with train data
    :param model: encoder-decoder NN
    :param metrics: dict of training stats
    :param optimizer: torch.nn.optim
    :param scheduler: LR scheduler
    :param gradient_accumulation_steps: perform optimizer step after this number of forward+loss.backward
    :param device:
    :return: None
    """
    for i, batch in tqdm(enumerate(train_loader), leave=False):
        batch.to(device)
        step = len(metrics['train_loss']) + 1
        optimizer.zero_grad()
        loss = _get_batch_loss(model, batch)
        loss.backward()
        if i % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
        metrics['train_loss'].append((step, loss.item()))


def val_step(val_loader, model, model_id, metrics, best_bleu, device):
    """
    :param val_loader: validation data loader
    :param model: encoder-decoder NN
    :param metrics: dict with training statistics
    :param best_bleu: previous best BLEU value
    :return: the best BLEU after validation step
    """
    mean_bleu = 0
    mean_loss = 0
    step = len(metrics['train_loss'])
    for batch in tqdm(val_loader, leave=False):
        with torch.no_grad():
            batch.to(device)
            loss, pred, tgt = _get_batch_loss(model, batch, return_pred=True)
            mean_bleu += np.sum([Evaluator.bleu_score(p, t) for p, t in zip(pred.cpu().numpy(), tgt.cpu().numpy())])
            mean_loss += loss.item()
    metrics['dev_loss'].append((step, mean_loss / len(val_loader)))
    metrics['dev_bleu'].append((step, 100 * mean_bleu / len(val_loader.dataset)))

    if best_bleu < metrics['dev_bleu'][-1][-1]:
        best_bleu = metrics['dev_bleu'][-1][-1]
        if isinstance(model, BaselineTranslationModel):
            torch.save(model.state_dict(), model_id)
        else:
            model.save_pretrained(model_id)

    return best_bleu


def test_step(test_loader, model, tokenizer, verbose, device):
    """
    Iterate over test_loader
    :param test_loader: test data loader
    :param model: encoder-decoder NN
    :param verbose: Is set to True, will log translation samples
    :return: None
    """
    evaluator = Evaluator()
    toxicity_drop = []
    similarity = []

    test_dataset = test_loader.dataset

    for i, batch in tqdm(enumerate(test_loader), leave=False):
        with torch.no_grad():
            batch.to(device)
            src_tokens, pred_tokens = _inference_model(model, batch, test_dataset, tokenizer)
            if (i % 50 == 0) and verbose:
                logger.info(f'Sample input: {src_tokens[0]}')
                logger.info(f'Sample prediction: {pred_tokens[0]}')
            similarity.extend(evaluator.estimate_similarity(pred_tokens, src_tokens).cpu().numpy().reshape(-1))
            src_toxicity = evaluator.estimate_toxicity(src_tokens, probs=True)
            pred_toxicity = evaluator.estimate_toxicity(pred_tokens, probs=True)
            toxicity_drop.extend((src_toxicity - pred_toxicity).reshape(-1))
    logger.info(f'Toxicity drop statistics: {lo(np.array(toxicity_drop))}')
    logger.info(f'Translation-Source similarity statistics: {lo(np.array(similarity))}')


def _train(epochs,
           model,
           tokenizer,
           model_id,
           optimizer,
           scheduler,
           gradient_accumulation_steps,
           train_loader,
           val_loader,
           test_loader,
           verbose_test,
           experiment_start,
           device
           ):
    """
    :param epochs: Number of epochs to train the model for
    :param model: encoder-decoder NN
    :param optimizer: torch.nn.optim
    :param scheduler: LR scheduler
    :param gradient_accumulation_steps: perform optimizer step after this number of forward+loss.backward
    :param train_loader: training data loader
    :param val_loader: validation data loader
    :param test_loader: test data loader
    :param verbose_test: If set to True, will print test set translation examples
    :param experiment_start: Time of script execution start
    :param device: device for model to be stored and trained on
    :return: None
    """
    metrics = {'train_loss': [], 'dev_bleu': [], 'dev_loss': []}
    best_bleu = 0.0

    for epoch in tqdm(range(epochs)):
        train_step(train_loader, model, metrics, optimizer, scheduler, gradient_accumulation_steps, device)
        best_bleu = val_step(val_loader, model, model_id, metrics, best_bleu, device)

        logger.info("Epoch %d Mean loss=%.3f" % (epoch, np.mean(metrics['train_loss'][-10:], axis=0)[1]))
        logger.info("Epoch %d Best BLEU=%.3f" % (epoch, best_bleu))

    plt.figure(figsize=(12, 4))
    for i, (name, history) in enumerate(sorted(metrics.items())):
        plt.subplot(1, len(metrics), i + 1)
        plt.title(name)
        plt.plot(*zip(*history))
        plt.grid()
    os.makedirs('reports/figures/', exist_ok=True)
    plt.savefig(f'reports/figures/{experiment_start}_baseline.png')

    test_step(test_loader, model, tokenizer, verbose_test, device)


def train_baseline(epochs: int,
                   batch_size: int = 32,
                   use_subword_tokenization: bool = False,
                   embeddings_size: int = 200,
                   hidden_size: int = 200,
                   n_layers: int = 3,
                   device_type: str = 'cuda:0',
                   verbose_test: bool = True
                   ):
    """
    :param epochs: Number of epochs to train the model for
    :param batch_size: dataloader's batch size
    :param use_subword_tokenization: If set to True, will use BPE
    :param embeddings_size: if set to 200 or 300, will load pre-trained GLoVE embeddings
    :param hidden_size: model's hidden size
    :param n_layers: number of layers in the recurrent unit
    :param device_type: where to store an train the model on
    :param verbose_test: If set to True, will print test set translation examples
    :return: None
    """
    device = torch.device(device_type)
    experiment_start = str(datetime.datetime.now()).replace(' ', '_')
    train_dataset = TextDetoxificationDataset(mode='train', use_bpe=use_subword_tokenization)
    val_dataset = TextDetoxificationDataset(mode='val', use_bpe=use_subword_tokenization, vocab=train_dataset.vocab)
    test_dataset = TextDetoxificationDataset(mode='test', use_bpe=use_subword_tokenization, vocab=train_dataset.vocab)

    def collate_batch(batch, max_len=64):
        source, target = [], []
        for src_sentence, tgt_sentence, _ in batch:
            source.append(torch.Tensor(
                [train_dataset.BOS_IDX] + src_sentence[:max_len].tolist() + [train_dataset.EOS_IDX]).long())
            target.append(torch.Tensor(
                [train_dataset.BOS_IDX] + tgt_sentence[:max_len].tolist() + [train_dataset.EOS_IDX]).long())

        source = torch.nn.utils.rnn.pad_sequence(source, batch_first=True, padding_value=train_dataset.PAD_IDX)
        target = torch.nn.utils.rnn.pad_sequence(target, batch_first=True, padding_value=train_dataset.PAD_IDX)

        return source.to(device), target.to(device)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=collate_batch)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                             collate_fn=collate_batch)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                              collate_fn=collate_batch)

    logger.info('Initializing model')
    model = BaselineTranslationModel(vocab=train_dataset.vocab,
                                     emb_dim=embeddings_size,
                                     hidden_dim=hidden_size,
                                     n_layers=n_layers).to(device)
    logger.info('Model initialized')

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    _train(epochs, model, None, 'models/baseline_best.pth', optimizer, scheduler, 1, train_loader, val_loader,
           test_loader, verbose_test,experiment_start, device)


def _prepare_t5_model_and_datasets(batch_size, virtual_tokens: int):
    train_df, val_df, test_df = [pd.read_csv(f'../data/interim/{stage}.tsv', sep='\t') for stage in
                                 ['train', 'val', 'test']]
    train_df = train_df.rename(columns={'reference': 'source', 'translation': 'target'})
    val_df = val_df.rename(columns={'reference': 'source', 'translation': 'target'})
    test_df = test_df.rename(columns={'reference': 'source', 'translation': 'target'})
    dataset = DatasetDict()
    dataset['train'] = Dataset.from_pandas(train_df[['source', 'target']])
    dataset['val'] = Dataset.from_pandas(val_df[['source', 'target']])
    dataset['test'] = Dataset.from_pandas(test_df[['source', 'target']])

    tokenizer = AutoTokenizer.from_pretrained(T5_CHECKPOINT)

    def preprocess_function(examples):
        model_inputs = tokenizer(examples['source'], max_length=T5_MAX_LENGTH - virtual_tokens, padding='max_length',
                                 truncation=True)
        labels = tokenizer(examples['target'], max_length=T5_MAX_LENGTH - virtual_tokens, padding='max_length',
                           truncation=True)
        labels[labels == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = dataset.map(preprocess_function,
                                     batched=True,
                                     remove_columns=dataset["train"].column_names,
                                     desc='preprocessing datasets')

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=T5_CHECKPOINT)
    train_loader = torch.utils.data.DataLoader(tokenized_datasets['train'], batch_size=batch_size, shuffle=True,
                                               collate_fn=data_collator)
    val_loader = torch.utils.data.DataLoader(tokenized_datasets['val'], batch_size=batch_size, shuffle=False,
                                             collate_fn=data_collator)
    test_loader = torch.utils.data.DataLoader(tokenized_datasets['test'], batch_size=batch_size, shuffle=False,
                                              collate_fn=data_collator)

    model = AutoModelForSeq2SeqLM.from_pretrained(T5_CHECKPOINT)

    return model, tokenizer, train_loader, val_loader, test_loader


def train_t5_lora(epochs: int = 5,
                  batch_size: int = 32,
                  gradient_accumulation_steps: int = 4,
                  lora_r: int = 8,
                  lora_alpha: int = 32,
                  lora_dropout: float = 0.1,
                  device_type: str = 'cuda:0',
                  verbose_test: bool = True):
    device = torch.device(device_type)
    experiment_start = str(datetime.datetime.now()).replace(' ', '_')

    model, tokenizer, train_loader, val_loader, test_loader = _prepare_t5_model_and_datasets(batch_size,
                                                                                             virtual_tokens=0)
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
        fan_in_fan_out=False,
    )
    for param in model.parameters():
        param.requires_grad = False

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    _train(epochs, model, tokenizer, 'models/T5-small-prefix-tuning-best', optimizer, scheduler,
           gradient_accumulation_steps, train_loader, val_loader, test_loader, verbose_test, experiment_start, device)


if __name__ == '__main__':
    # Examples
    # python src/models/train_model.py baseline --epochs=10 --batch_size=32 --embeddings_size=200
    # python src/models/train_model.py t5_lora --epochs=5 --batch_size=32 --accumulate_steps=4

    sys.path.append(os.getcwd())
    torch.manual_seed(0)
    random.seed(0)

    from src.data.make_dataset import TextDetoxificationDataset, Evaluator

    Fire(
        {
            "baseline": train_baseline,
            "t5_lora": train_t5_lora,
            "t5_prefix_tuning": None
        }
    )
