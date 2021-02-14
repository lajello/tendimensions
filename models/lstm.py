import torch
from torch import nn
import torch.nn.functional as F
import os

class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(LSTMClassifier, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.lstm = nn.LSTM(embedding_dim, hidden_dim,batch_first=True)
        self.W_out = nn.Linear(hidden_dim,1)

    def forward(self, batch):
        """

        :param batch of size [b @ (seq x dim)]
        :return: array of size [b]
        """
        lengths = (batch!=0).sum(1)[:,0] # lengths of non-padded items
        lstm_outs, _ = self.lstm(batch) # [b x seq x dim]
        out = torch.stack([lstm_outs[i,idx-1] for i,idx in enumerate(lengths)])
        out = self.W_out(out).squeeze()
        # out = torch.sigmoid(out).squeeze()

        return out


# os.chdir('../')
# from features.embedding_features import ExtractWordEmbeddings
# em = ExtractWordEmbeddings('glove',
#                            emb_dir='/10TBdrive/minje/features/embeddings')