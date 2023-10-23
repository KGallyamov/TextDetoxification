import sys
import os
import datetime

import gensim.downloader as api
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from fire import Fire
from loguru import logger
from lovely_numpy import lo
from torch import nn
from tqdm.auto import tqdm

GLOVE_WIKI_EMBEDDING_DIM = 300
GLOVE_TWITTER_EMBEDDING_DIM = 200


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


def _compute_loss(logits_seq, out, pad_idx, vocab_size):
    mask = out != pad_idx  # [batch_size, out_len]
    targets_1hot = F.one_hot(out, vocab_size).to(torch.float32)

    # log-probabilities of all tokens at all steps, [batch_size, out_len, num_tokens]
    logprobs_seq = torch.log(logits_seq.softmax(dim=-1))

    # log-probabilities of correct outputs, [batch_size, out_len]
    logp_out = (logprobs_seq * targets_1hot).sum(dim=-1)

    # average cross-entropy over non-padding tokens
    return - torch.masked_select(logp_out, mask).mean()


def train_step(train_loader, model, metrics, optimizer, scheduler):
    for src, tgt in tqdm(train_loader, leave=False):
        step = len(metrics['train_loss']) + 1
        optimizer.zero_grad()
        loss = _compute_loss(model(src, tgt), tgt, model.vocab['<pad>'], len(model.vocab))
        loss.backward()
        optimizer.step()
        scheduler.step()
        metrics['train_loss'].append((step, loss.item()))


def val_step(val_loader, model, metrics, best_bleu):
    mean_bleu = 0
    mean_loss = 0
    step = len(metrics['train_loss'])
    for src, tgt in tqdm(val_loader, leave=False):
        with torch.no_grad():
            pred = model(src, tgt)
            loss = _compute_loss(pred, tgt, model.vocab['<pad>'], len(model.vocab))
            pred = pred.argmax(-1).cpu().numpy()
            mean_bleu += np.sum([Evaluator.bleu_score(p, t) for p, t in zip(pred, tgt.cpu().numpy())])
            mean_loss += loss.item()
    metrics['dev_loss'].append((step, mean_loss / len(val_loader)))
    metrics['dev_bleu'].append((step, 100 * mean_bleu / len(val_loader.dataset)))

    if best_bleu < metrics['dev_bleu'][-1][-1]:
        best_bleu = metrics['dev_bleu'][-1][-1]
        torch.save(model.state_dict(), 'models/baseline_best.pth')

    return best_bleu


def test_step(test_loader, test_dataset, model, verbose: bool = True):
    evaluator = Evaluator()
    toxicity_drop = []
    similarity = []

    for i, (src, tgt) in tqdm(enumerate(test_loader), leave=False):
        with torch.no_grad():
            encoded = model.encode(src)
            pred, _ = model.decode_inference(encoded)
            pred_tokens = test_dataset.detokenize(pred)
            src_tokens = test_dataset.detokenize(src)
            if (i % 50 == 0) and verbose:
                logger.info(f'Sample input: {src_tokens[0]}')
                logger.info(f'Sample prediction: {pred_tokens[0]}')
            similarity.extend(evaluator.estimate_similarity(pred_tokens, src_tokens).cpu().numpy().reshape(-1))
            src_toxicity = evaluator.estimate_toxicity(src_tokens, probs=True)
            pred_toxicity = evaluator.estimate_toxicity(pred_tokens, probs=True)
            toxicity_drop.extend((src_toxicity - pred_toxicity).reshape(-1))
    logger.info(f'Toxicity drop statistics: {lo(np.array(toxicity_drop))}')
    logger.info(f'Translation-Source similarity statistics: {lo(np.array(similarity))}')


def train_baseline(epochs: int,
                   batch_size: int = 32,
                   embeddings_size: int = 200,
                   hidden_size: int = 200,
                   n_layers: int = 3,
                   device_type: str = 'cuda:0',
                   verbose_test: bool = True
                   ):
    device = torch.device(device_type)
    experiment_start = str(datetime.datetime.now()).replace(' ', '_')
    train_dataset = TextDetoxificationDataset(mode='train')
    val_dataset = TextDetoxificationDataset(mode='val', vocab=train_dataset.vocab)
    test_dataset = TextDetoxificationDataset(mode='test', vocab=train_dataset.vocab)

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

    metrics = {'train_loss': [], 'dev_bleu': [], 'dev_loss': []}  # 'dev_tox_drop': [], 'dev_sim': []}
    best_bleu = 0.0

    for epoch in tqdm(range(epochs)):
        train_step(train_loader, model, metrics, optimizer, scheduler)
        best_bleu = val_step(val_loader, model, metrics, best_bleu)

        logger.info("Epoch %d Mean loss=%.3f" % (epoch, np.mean(metrics['train_loss'][-10:], axis=0)[1]))
        logger.info("Epoch %d Best BLEU=%.3f" % (epoch, best_bleu))

    plt.figure(figsize=(12, 4))
    for i, (name, history) in enumerate(sorted(metrics.items())):
        plt.subplot(1, len(metrics), i + 1)
        plt.title(name)
        plt.plot(*zip(*history))
        plt.grid()
    plt.savefig(f'reports/figures/{experiment_start}_baseline.png')

    test_step(test_loader, test_dataset, model, verbose_test)


if __name__ == '__main__':
    # python src/models/train_model.py baseline --epochs=10
    sys.path.append(os.getcwd())

    from src.data.make_dataset import TextDetoxificationDataset, Evaluator

    Fire(
        {
            "baseline": train_baseline,
            "T5_soft": None,
            "T5_prefix": None
        }
    )
