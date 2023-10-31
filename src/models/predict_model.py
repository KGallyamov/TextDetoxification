import os
import random
import sys

import torch
from fire import Fire
from loguru import logger


# possible simplification: create training configs and load both train and inference settings from there
def inference_baseline(source: str,
                       vocab=None,
                       checkpoint_path: str = 'models/rnn_best.pth',
                       use_subword_tokenization: bool = False,
                       embeddings_size: int = 200,
                       hidden_size: int = 200,
                       n_layers: int = 3,
                       device_type: str = 'cuda:0', ):
    """
    :param source: Text to be detoxified
    :param vocab: Path to the vocab build during model training (if not provided, build the vocab from train data)
    :param checkpoint_path: Path at which pretrained model weights are stored
    :param use_subword_tokenization: Set to True if the same parameter was True during training
    :param embeddings_size: Embeddings dim in the pretrained model
    :param hidden_size: Hidden size of the pretrained model
    :param n_layers: Number of layers in the original (trained) model
    :param device_type: str, device on which to run the model
    :return: None, log the translation into console
    """
    # suboptimal solution so far: create base dataset every time this function is called
    # possible solution - save the vocab during model training
    device = torch.device(device_type)
    if vocab is not None:
        vocab = torch.load(vocab)
    base_dataset = TextDetoxificationDataset('train', use_bpe=use_subword_tokenization, vocab=vocab)
    logger.info('Loading model')
    model = BaselineTranslationModel(vocab=base_dataset.vocab, emb_dim=embeddings_size, hidden_dim=hidden_size,
                                     n_layers=n_layers, init_embeddings_if_possible=False)
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    model.eval()
    logger.info('Model loaded')
    tokens = base_dataset.tokenize_sentence(source)
    tokens_indices = base_dataset.vocab(tokens)
    tokens_tensor = torch.tensor([tokens_indices], dtype=torch.int64, device=device)
    logger.info('Inferencing the sentence')
    with torch.no_grad():
        encoded = model.encode(tokens_tensor)
        decoded = model.decode_inference(encoded)[0].cpu().numpy()

    decoded_tokens = base_dataset.detokenize(decoded)[0]
    logger.info(f'Translation: {decoded_tokens}')


def inference_t5(source, peft_config_path, top_p: float = 0.8, device_type: str = 'cuda:0', ):
    """
    :param source: Text to be detoxified
    :param peft_config_path: path at which the pretrained model is stored
    :param top_p: P in nucleus sampling
    :param device_type: str, device on which to run the model
    :return: None, log the translation into console
    """
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from peft import PeftConfig, PeftModel

    device = torch.device(device_type)

    tokenizer = AutoTokenizer.from_pretrained(T5_CHECKPOINT)

    logger.info('Loading model')
    config = PeftConfig.from_pretrained(peft_config_path)

    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path, device_map={"": 0})
    model = PeftModel.from_pretrained(model, peft_config_path)
    model.eval()
    logger.info(f'Model loaded with config: {config}')

    batch = tokenizer(source, max_length=T5_MAX_LENGTH, padding='max_length', truncation=True)
    logger.info('Inferencing model')
    with torch.no_grad():
        input_ids = torch.tensor([batch['input_ids']], dtype=torch.int64, device=device)
        attention_mask = torch.tensor([batch['attention_mask']], dtype=torch.int64, device=device)
        outputs = model.generate(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 do_sample=True,
                                 top_p=top_p,
                                 max_length=T5_MAX_LENGTH)
    pred_tokens = tokenizer.batch_decode(outputs.cpu().numpy(), skip_special_tokens=True)[0]
    logger.info(f'Translation: {pred_tokens}')


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    torch.manual_seed(0)
    random.seed(0)

    from src.models.train_model import BaselineTranslationModel, T5_CHECKPOINT, T5_MAX_LENGTH
    from src.data.make_dataset import TextDetoxificationDataset

    Fire(
        {
            "baseline": inference_baseline,
            "t5": inference_t5,
        }
    )
