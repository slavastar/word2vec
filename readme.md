# Word2Vec

PyTorch implementation of the Word2Vec paper: [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf)

Most of the code was taken from [this repository](https://github.com/OlgaChernytska/word2vec-pytorch/tree/66ba7adabb56905185e95a00ef0089c6d16dda49).

## Description
The Word2Vec model was trained on [Wiki2](https://paperswithcode.com/dataset/wikitext-2) dataset.
The hyperparameters of the model can be found in the `config.yaml` file.

## Project structure

* `model` - folder with implemented models of 2 types: CBOW and Skip-Gram.
* `config.yaml` - config file with the main parameters of the model.
* `dataloader.py` - contains functions for text preprocessing and collecting a dataset.
* `demo.ipynb` - demo notebook with several examples of model inference with visualisation.
* `train.py` - contains a full pipeline for training and saving the model.
* `training.py` - contains a wrapper for training a model.
* `utils.py` - contains common functions used by other modules.


## Usage
```
python train.py --config config.yaml
```