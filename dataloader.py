from functools import partial

import torch
from torch.utils.data import DataLoader
from torchtext.datasets import WikiText2
from torchtext.data import to_map_style_dataset, get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


def get_english_tokenizer():
    return get_tokenizer('basic_english', language='en')


def get_data_iter(data_dir: str, split = ('valid')):
    data_iter = WikiText2(root=data_dir, split=split)
    return to_map_style_dataset(data_iter)


def build_vocab(data_iter, tokenizer, min_word_frequency: int = 50):
    vocab = build_vocab_from_iterator(
        map(tokenizer, data_iter),
        specials=['<unk>'],
        min_freq=min_word_frequency
    )
    vocab.set_default_index(vocab['<unk>'])
    return vocab


def collate_cbow(batch: list[str], text_pipeline, n_words: int, max_sequence_length: int = None):
    batch_input, batch_output = [], []
    for text in batch:
        text_tokens_ids = text_pipeline(text)

        if len(text_tokens_ids) < 2 * n_words + 1:
            continue

        if max_sequence_length is not None:
            text_tokens_ids = text_tokens_ids[:max_sequence_length]

        for index in range(len(text_tokens_ids) - 2 * n_words):
            sequence = text_tokens_ids[index:(index + 2 * n_words + 1)]
            sequence_output = sequence.pop(n_words)
            sequence_input = sequence
            batch_input.append(sequence_input)
            batch_output.append(sequence_output)

    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)

    return batch_input, batch_output


def collate_skip_gram(batch: list[str], text_pipeline, n_words: int, max_sequence_length: int = None):
    batch_output, batch_input = collate_cbow(batch, text_pipeline, n_words, max_sequence_length)
    return batch_input, batch_output


def get_dataloader_and_vocab(data_dir: str, split: str = 'train', model_name: str = 'CBOW', batch_size: int = 64,
                             shuffle: bool = True, min_word_frequency: int = 50, n_words: int = 5,
                             max_sequence_length: int = None):
    data_iter = get_data_iter(data_dir, split)
    tokenizer = get_english_tokenizer()
    vocab = build_vocab(data_iter, tokenizer, min_word_frequency)

    if model_name == 'CBOW':
        collate_fn = collate_cbow
    elif model_name == 'SkipGram':
        collate_fn = collate_skip_gram
    else:
        raise ValueError(f'model_name can be either "CBOW" or "SkipGram". Got {model_name} instead')

    dataloader = DataLoader(
        dataset=data_iter,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_fn, text_pipeline=lambda x: vocab(tokenizer(x)),
                           n_words=n_words, max_sequence_length=max_sequence_length)
    )

    return dataloader, vocab
