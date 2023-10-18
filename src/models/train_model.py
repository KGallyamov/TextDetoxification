import gensim.downloader as api
import torch
import torch.nn.functional as F
from fire import Fire
from torch import nn
from tqdm.auto import tqdm

GLOVE_WIKI_EMBEDDING_DIM = 300
GLOVE_TWITTER_EMBEDDING_DIM = 200
t = torch.optim.lr_scheduler.CosineAnnealingLR


class BaselineTranslationModel(nn.Module):
    # This code is adapted from:
    # https://github.com/yandexdataschool/nlp_course/blob/2022/week04_seq2seq/practice_and_homework_pytorch.ipynb

    def __init__(self, vocab, emb_dim, hidden_dim, n_layers):
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
        outputs = [torch.full([batch_size], self.vocab['<bos>'], dtype=torch.int64,
                              device=device)]
        all_states = [initial_state]
        for i in range(max_len):
            state, logits = self.decode_step(state, outputs[-1])
            probs = F.softmax(logits, -1)
            if top_p == -1:
                outputs.append(logits.argmax(-1))
                all_states.append(state)
                continue

            output = torch.zeros((batch_size,))
            for b in range(batch_size):  # nucleus sampling
                token_probs = probs[b]
                s_probs, indices = torch.sort(token_probs, descending=True, dim=0)
                idx = int((torch.cumsum(s_probs, dim=0) > top_p).nonzero())
                token_idx = torch.multinomial(s_probs[:idx + 1] / s_probs[:idx + 1].sum(), num_samples=1)
                output[idx] = indices[token_idx]
            outputs.append(output)
            all_states.append(state)

        return torch.stack(outputs, dim=1), all_states


def train(model_type: str, steps: int, optimizer: str, scheduler: str):
    assert model_type.lower() in {'baseline', 'experiment1', 'experiment2'}


if __name__ == '__main__':
    Fire(train)
