# This file Train a word Embedding Using NNLM
# Implementing refer to official tutorial of pytorch
# https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext.datasets import WikiText2
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from tensorboardX import SummaryWriter

# download WikiText2
# WikiText2.download('./corpus')

class WikiText2DataSet(Dataset):
    """
    We are just training word embeddings, what we need is just text,
    And thus we do not perform train, val, test splitting and sort of
    things. You can change the data file to whatever you want as long
    as it's plain text, and it's not that big.
    It's toy implementation, train on rather small dataset,
    so we don't restrict vocabulary size.
    """
    def __init__(self, data_file_path, ngram=5):
        """
        :param data_file_path: path for the plain text file
        :param ngram:  language model n-grams
        """
        with open(data_file_path,'r',encoding='utf-8') as f:
            s = f.read().lower()
        words_tokenized = word_tokenize(s)
        self.grams =  [([words_tokenized[i+j] for j in range(ngram-1)], words_tokenized[i + ngram -1])
            for i in range(len(words_tokenized) - ngram + 1)]
        self.vocab = Counter(words_tokenized)
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab.keys())}
        self.vocab_size = len(self.vocab)
        self.ngram = ngram

    def __getitem__(self, idx):
        context = torch.tensor([self.word_to_idx[w] for w in self.grams[idx][0]])
        target = torch.tensor([self.word_to_idx[self.grams[idx][1]]])
        return context, target

    def __len__(self):
        return len(self.grams)

class NGramLanguageModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, ngram):
        super(NGramLanguageModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear((ngram - 1) * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs, batch_size):

        embeds = self.embeddings(inputs).view((batch_size, -1)) # concat the word representations [batch_size, (ngram - 1) x embedding]
        out = F.relu(self.linear1(embeds)) # nonlinear + projection
        out = self.linear2(out) # linear project to vocabulary space
        log_probs = F.log_softmax(out, dim=1) # softmax compute log probability
        return log_probs


NGRAM = 5
EMBEDDING_DIM = 50
BATCH_SIZE = 10
NUM_EPOCH = 20

# I think torchtext is really hard to use
# It's a toy example, so you can use any plain text dataset
# data_file_path = './corpus/wikitext-2/wikitext-2/wiki.train.tokens'
data_file_path = './corpus/Pride-and-Prejudice.txt'

data = WikiText2DataSet(data_file_path=data_file_path)
model = NGramLanguageModel(len(data.vocab), EMBEDDING_DIM, NGRAM)
# optimizer = optim.SGD(model.parameters(), lr=0.001)
optimizer = optim.Adam(model.parameters())
loss_function = nn.NLLLoss()
losses = []
cuda_available = torch.cuda.is_available()
data_loader = DataLoader(data, batch_size=BATCH_SIZE)

# Writer
writer = SummaryWriter('./logs/NNLM')

for epoch in range(NUM_EPOCH):
    total_loss = 0
    for context, target in tqdm(data_loader):
        # context: torch.Size([10, 4])
        # target:  torch.Size([10, 1])
        if context.size()[0] != 10: continue
        # deal with last several batches

        if cuda_available:
            context = context.cuda()
            target = target.squeeze(1).cuda()
            model = model.cuda()

        model.zero_grad()
        log_probs = model(context, BATCH_SIZE)
        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    losses.append(total_loss)
    writer.add_scalar('Train/Loss', total_loss, epoch)
    print('total_loss:',total_loss)
