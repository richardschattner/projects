from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
import pandas as pd
import tiktoken


@dataclass
class modelconfig:
    #gpt
    block_size :int = 256
    vocab_size :int = 50258 #gpt2 tokenizer vocab size +1 special token for padding
    n_layer :int = 2
    n_head :int = 2
    n_embd :int = 64
    n_lm_head :int = 32
    #help model
    n_help : int = 4
    #final model
    n1 :int = 16
    n2 :int = 8
    n3 :int = 8



#model
class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gpt = GPT(config)
        self.help_model = nn.Sequential(
            nn.Linear(2, config.n_help),
            nn.Tanh(),
            nn.Linear(in_features= config.n_help ,out_features= config.n_help),
            nn.LayerNorm(config.n_help),
        )

        self.final = nn.Sequential(
            nn.Linear(in_features= config.n_lm_head + config.n_help, out_features= config.n1),
            nn.ReLU(),
            nn.Linear(config.n1, config.n2),
            nn.Tanh(),
            nn.Linear(config.n2, 1)
            )

    def init_weights(self):
        self.gpt.init_weights()

    def forward(self, x, y):
        #forward pass for both submodels
        x1 = self.gpt(x)
        x2 = self.help_model(y)
        x = torch.cat([x1, x2], dim = 1)
        #forward pass for final model
        x = self.final(x)
        return x



#transformer model to analyze the review text, outputs n_embd
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), #self attention block
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.n_lm_head)

    #TODO
    def init_weights(self):
        pass

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(0, T, dtype = torch.long)
        pte = self.transformer.wpe(pos)
        wte = self.transformer.wte(x)
        x = wte + pte
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x[:, -1, :]) #(B,T,C) -> (B,C)
        x = self.lm_head(x)
        return x


#attention block
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.h = nn.ModuleList(config.n_layer * [
            nn.LayerNorm(config.n_embd), #we put the layernorm before the attention heads
            AttentionHead(config),
            nn.LayerNorm(config.n_embd),
            AttentionHead(config),
            nn.Linear(config.n_embd, config.n_embd)
        ])

    def forward(self, x):
        for layer in self.h:
            x = layer(x)
        return x


class AttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()  
        assert config.n_embd % config.n_head == 0
        self.attn = nn.Linear(config.n_embd, 3* config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.config = config

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.attn(x)
        #split into (B, n_heads, T, head_size), where head_size = n_embd / n_head
        q, k, v = qkv.split(self.config.n_embd, dim=2)
        q = q.view(B, T, self.config.n_head, self.config.n_embd // self.config.n_head ).transpose(1, 2)
        k = k.view(B, T, self.config.n_head, self.config.n_embd // self.config.n_head ).transpose(1, 2)
        v = v.view(B, T, self.config.n_head, self.config.n_embd // self.config.n_head ).transpose(1, 2)

        #(unmasked) flash attention
        y = F.scaled_dot_product_attention(q, k, v) #(B, n_head, T, head_size)
        #reassemble the outputs into format (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        return y

#---------------------------------------------------------------------------------------------------------------------------------
#Data pre-processing and definining the dataloader

#load dataset
data = pd.read_csv("Reviews.csv")
data = data[["Text", "HelpfulnessNumerator", "HelpfulnessDenominator", "Score"]]

#encode the text
enc = tiktoken.get_encoding("gpt2")
tokens = []
for i in range(len(data)):
    tokens.append(enc.encode(data.loc[i, "Text"]))
data["Text"] = tokens

#add a column containing the amount of tokens of each review
for i in range(len(data)):
    data.loc[i, "length"] = len(data.loc[i, "Text"])

#this allows us to filter out reviews which are too long.
data = data[data["length"] <= 256]

#Adding a special token to the vocabulary. This token will be used to pad reviews to fit the block size.
pad_token = 50257


class DataLoader():
    def __init__(self, data, batch_size = 32, block_size = 256):
        self.iteration = 0
        self.data = data
        self.batch_size = batch_size
        self.block_size = block_size
        self.n_batches = len(data)// self.batch_size if len(data) % self.batch_size == 0 else len(data)// self.batch_size + 1 

    def prepare_batch(self):
        #normal batch
        idx = self.batch_size * self.iteration
        if (self.iteration + 1 ) != self.n_batches:
            self.iteration += 1
            return self.data[idx : idx + self.batch_size]
        #last batch
        else: 
            #reset the iteration counter
            self.iteration = 0
            return self.data[idx:]
    
    def get_batch(self):
        raw_batch = self.prepare_batch()
        x1 = torch.tensor(raw_batch["HelpfulnessNumerator"])
        x2 = torch.tensor(raw_batch["HelpfulnessDenominator"])
        x = torch.stack((x1, x2) , dim = 0)
        target = torch.tensor(raw_batch["Score"])

        #take first row
        #the tokenized text is of variable length and must be padded to fit the transformers block size
        row_token = raw_batch.loc[0 , "Text"]
        row_token += [1] * (self.block_size - len(row_token))
        y = torch.tensor(row_token, dtype= torch.long).unsqueeze(0)

        #repeat for other rows
        #the tokenized text is of variable length and must be padded to fit the transformers block size
        for j in range(self.batch_size - 1):
            row_token = raw_batch.loc[j + 1, "Text"]
            row_token += [pad_token] * (self.block_size - len(row_token))
            y = torch.cat((y, torch.tensor(row_token, dtype= torch.long).unsqueeze(0)), dim = 0) 
                    
        return x, y, target

#---------------------------------------------------------------------------------------------------------------------------------
#training and evaluation functions

def train(model, optimizer, dataloader, loss_fn):
    x, y, target = dataloader.get_batch()
    optimizer.zero_grad()
    pred = model(x, y)
    loss = loss_fn(target, pred)
    optimizer.step()
    return loss.detach().item()

@torch.no_grad()
def validate(model, dataloader, loss_fn):
    x, y, target = dataloader.get_batch()
    pred = model(x, y)
    loss = loss_fn(target, pred)
    return loss.detach().item()

def fit(model, iterations, optimizer, train_loader, val_loader, loss_fn):
    for i in range(iterations):
        train_loss = train(model, optimizer, train_loader, loss_fn)

        #validate and print losses every 100 iterations
        if i % 100 == 0:
            val_loss = validate(model, val_loader, loss_fn)
            print(f"Iterations * 100: {i} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f} ")


#---------------------------------------------------------------------------------------------------------------------------------
#initializing the model, loss function and optimizer
model = Model(modelconfig)
loss_fn = F.l1_loss()
optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4)

#dataloader
split1, split2 = int(0.8 * len(data)), int(0.1 * len(data))
#split data
train_set = data[:split1]
val_set = data[split1 : split2]
test_set = data[split2:]
#init dataloaders
train_loader = DataLoader(train_set)
val_loader = DataLoader(val_set)
test_loader = DataLoader(test_set)

#---------------------------------------------------------------------------------------------------------------------------------
#training the model



x = torch.tensor(enc.encode("hello world"), dtype = torch.long).unsqueeze(0)
y = torch.tensor([1,2] ,dtype= torch.float).unsqueeze(0)
print(model(x, y))
