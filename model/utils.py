from word2vec.model import CBOW, SkipGram


def get_model_class(name: str):
    if name == 'CBOW':
        return CBOW
    elif name == 'SkipGram':
        return SkipGram
    raise ValueError(f'Model class can be either "CBOW" or "SkipGram. Got {name} instead.')
