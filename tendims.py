import os
import sys
from os.path import join

import nltk
import numpy as np
import torch
from nltk.tokenize import TweetTokenizer

from .features.embedding_features import ExtractWordEmbeddings
from .models.lstm import LSTMClassifier

tokenize = TweetTokenizer().tokenize
from nltk import sent_tokenize

nltk.download("punkt")

dimensions_list = [
    "support",
    "knowledge",
    "conflict",
    "power",
    "similarity",
    "fun",
    "status",
    "trust",
    "identity",
    "romance",
]


class TenDimensionsClassifier:
    def __init__(
        self,
        models_dir="./models/lstm_trained_models",
        embeddings_dir="./embeddings",
        is_cuda=False,
    ):
        """
        @param models_dir: the directory where the LSTM models are stored
        @param embeddings_dir: the directory where the embeddings are stored. The directory must contain the following subdirectories:
                               word2vec/GoogleNews-vectors-negative300.wv
                               fasttext/wiki-news-300d-1M-subword.wv
                               glove/glove.42B.300d.wv
        @param is_cuda: to enable cuda
        """
        self.is_cuda = is_cuda
        self.models_dir = models_dir
        self.embeddings_dir = embeddings_dir

        # load embeddings
        self.em_glove = ExtractWordEmbeddings("glove", emb_dir=self.embeddings_dir)
        self.em_word2vec = ExtractWordEmbeddings(
            "word2vec", emb_dir=self.embeddings_dir
        )
        self.em_fasttext = ExtractWordEmbeddings(
            "fasttext", emb_dir=self.embeddings_dir
        )
        self.dimensions_list = [
            "support",
            "knowledge",
            "conflict",
            "power",
            "similarity",
            "fun",
            "status",
            "trust",
            "identity",
            "romance",
        ]

        # load models
        self.dim2model = {}
        self.dim2embedding = {}

        for dim in self.dimensions_list:
            model = LSTMClassifier(embedding_dim=300, hidden_dim=300)
            if self.is_cuda:
                print(f"Torch version: {torch.__version__}")
                print(f"Torch CUDA available : {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    print(f"Torch current device : {torch.cuda.current_device()}")
                    print(f"Torch device count : {torch.cuda.device_count()}")
                    print(f"Torch device name : {torch.cuda.get_device_name(0)}")
                    model.cuda()
                else:
                    print(
                        "Cuda not available. Instantiated the TenDimensionsClassifier with CUDA=False"
                    )
                    self.is_cuda = False
            model.eval()
            for modelname in os.listdir(self.models_dir):
                if ("-best.lstm" in modelname) & (dim in modelname):
                    best_state = torch.load(
                        join(self.models_dir, modelname), map_location="cpu"
                    )
                    model.load_state_dict(best_state)
                    if "glove" in modelname:
                        em = self.em_glove
                    elif "word2vec" in modelname:
                        em = self.em_word2vec
                    elif "fasttext" in modelname:
                        em = self.em_fasttext
                    self.dim2model[dim] = model
                    self.dim2embedding[dim] = em
                    break

    def _parse_input_dimensions(self, d):
        if d is None:
            return self.dimensions_list
        elif isinstance(d, str):
            return [d]
        elif isinstance(d, list) or isinstance(d, tuple):
            return d
        else:
            raise Exception(
                "Unrecognized input for dimension or dimension list: %s" % d
            )

    def compute_score(self, text, dimensions=None):
        """
        Computed dimension(s) scores on the whole input text
        @param text: the input text
        @param dimensions: a string representing the dimension or a list of strings for
                           multiple dimensions. None triggers the computation of all dimensions
        @return the confidence score for the selected dimension
                a dictionary dimension:score is returned if multiple dimensions were specified
                None (or dimension:None) is returned when the dimension could not be computed
        """
        dimension_scores = {d: None for d in self._parse_input_dimensions(dimensions)}
        if text is not None and text != "":
            for dim in dimension_scores:
                try:
                    model = self.dim2model[dim]
                    em = self.dim2embedding[dim]
                    input_ = em.obtain_vectors_from_sentence(tokenize(text), True)
                    input_ = torch.tensor(input_).float().unsqueeze(0)
                    if self.is_cuda:
                        input_ = input_.cuda()
                    output = model(input_)
                    score = torch.sigmoid(output).item()
                    dimension_scores[dim] = score
                except:
                    pass
        if len(dimension_scores) == 1:
            return list(dimension_scores.values())[0]
        else:
            return dimension_scores

    def compute_score_split(self, text, dimensions=None, min_tokens=3):
        """
        Computed dimension(s) scores on each sentence of the input text and returns aggreagated
        stats (avg and max)
        @param text: the input text
        @param dimensions: a string representing the dimension or a list of strings for
                           multiple dimensions. None triggers the computation of all dimensions
               min_tokens: the minimum number of tokens in a sentence for the dimension to be computed
        @return a tuple (avg, max, min, std) of confidence scores for the selected dimension
                a dictionary dimension:(avg, max, min, std) is returned if multiple dimensions were specified
                None (or dimension:None) is returned when the dimension could not be computed
        """
        dimension_scores = {
            d: (None, None) for d in self._parse_input_dimensions(dimensions)
        }
        if text is not None and text != "":
            for dim in dimension_scores:
                scores = []
                try:
                    sentences = sent_tokenize(text)
                except:
                    sentences = [text]

                for sent in sentences:
                    sent = str(sent)
                    if (
                        sent is not None
                        and sent != ""
                        and sent != "nan"
                        and len(sent.split()) >= min_tokens
                    ):
                        try:
                            score = self.compute_score(sent, dim)
                            if score is not None:
                                scores.append(score)
                        except:
                            pass
                if scores:
                    dimension_scores[dim] = (
                        np.mean(scores),
                        np.max(scores),
                        np.min(scores),
                        np.std(scores),
                    )
                else:
                    dimension_scores[dim] = (None, None, None, None)
        if len(dimension_scores) == 1:
            return list(dimension_scores.values())[0]
        else:
            return dimension_scores
