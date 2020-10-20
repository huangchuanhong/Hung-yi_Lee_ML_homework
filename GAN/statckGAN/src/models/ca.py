import torch
import torch.nn as nn

class CA(nn.Module):
    def __init__(self, embedding_dim, condition_dim):
        super(CA, self).__init__()
        self.linear = nn.Linear(embedding_dim, condition_dim * 2)
        self.relu = nn.ReLU()
        self.embedding_dim = embedding_dim
        self.condition_dim = condition_dim

    def get_mu_logvar(self, embeddings):
        outputs = self.linear(embeddings)
        outputs = self.relu(outputs)
        # outputs = self.relu(self.linear(embeddings))
        mu = outputs[:, :self.condition_dim]
        logvar = outputs[:, self.condition_dim:]
        return mu, logvar

    def forward(self, embeddings):
        mu, logvar = self.get_mu_logvar(embeddings)
        std = (logvar * 0.5).exp()
        eps = torch.zeros(mu.size(), device=embeddings.device).normal_()
        c = mu + eps * std
        return c, mu, logvar
