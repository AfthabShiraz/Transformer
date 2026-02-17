import torch
import torch.nn as nn
from torch.nn import functional as F

with open('/Users/afthabshiraz/Documents/Transformer/input.txt', 'r',encoding='utf-8') as f:
  text=f.read()


#hyperparameters
torch.manual_seed(1337)
batch_size=64 #how many sequences processed in parallel
block_size=256 
max_iters=5000
eval_interval=300
learning_rate=3e-4
eval_iters=200
n_layers=6
n_embd=384
dropout=0.2
num_heads=6
head_size=n_embd//num_heads
device='cuda' if torch.cuda.is_available() else 'cpu'


chars=sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] #encoder
decode = lambda l: ''.join([itos[i] for i in l]) #decoder

data = torch.tensor(encode(text),dtype=torch.long)
n= int(0.9*len(data))
train_data=data[:n]
val_data=data[n:]

block_size=8 #a chunk of 9 character has 8 examples
x=train_data[:block_size]
y=train_data[1:block_size+1] 

def get_batch(split):
  #generate a batch of data 
  data = train_data if split =='train' else val_data 
  """
  below line chooses 4 random indexes data while ensuring we can complete
  a batch from each index
  """
  ix = torch.randint(len(data) - block_size, (batch_size,)) 
  """
  below creates a  two 4 by 8 tensors. y is just shifted by 1.
  """
  x= torch.stack([data[i:i+block_size] for i in ix])
  y= torch.stack([data[i+1:i+block_size+1] for i in ix])
  x,y=x.to(device),y.to(device)
  return x,y

xb,yb=get_batch('train')
print('inputs: ')
print(xb.shape)
print(xb)
print('targets: ')
print(yb.shape)
print(yb)
print('----------')

for b in range(batch_size):
  for t in range(block_size):
    context=xb[b,:t+1]
    target=yb[b,t]
    print(f"when input is {context.tolist()} the target: {target}")

@torch.no_grad() #no need to compute gradients. better for memory usage
def estimate_loss():
  out={}
  model.eval()
  for split in ['train','val']:
    losses=torch.zeros(eval_iters)
    for k in range(eval_iters):
      X,Y=get_batch(split)
      logits,loss=model(X,Y)
      losses[k]=loss.item()
    out[split]=losses.mean()
  model.train()
  return out

class Head(nn.Module):
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout)
    
  def forward(self, x):
    B,T,C = x.shape
    k = self.key(x) #(B,T,C)
    q = self.query(x) #(B,T,C)
    v = self.value(x) #(B,T,C)
    wei = q @ k.transpose(-2,-1) * C**-0.5
    wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))#makes it a decoder
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)
    out = wei @ v # aggregate the values
    return out

class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(n_embd, n_embd)
    self.dropout = nn.Dropout(dropout)
  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.proj(out)
    return out

class FeedForward(nn.Module):
  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, 4 * n_embd),
      nn.ReLU(),
      nn.Linear(4 * n_embd, n_embd),
      nn.Dropout(dropout)
    )
  def forward(self, x):
    return self.net(x)

class Block(nn.Module):
  def __init__(self):
    super().__init__()
    self.sa = MultiHeadAttention(num_heads, head_size)
    self.ffwd = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd) 
    self.ln2 = nn.LayerNorm(n_embd) 
  def forward(self, x):
    x = x+ self.sa(self.ln1(x))
    x = x+ self.ffwd(self.ln2(x))
    return x

class BigramLanguageModel(nn.Module):

  def __init__(self, vocab_size) -> None:
    super().__init__()
    #each token directly reads off the logits for the next token from a lookup table
    #
    self.token_embedding_table =nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table =nn.Embedding(block_size, n_embd)
    self.blocks = nn.Sequential(*[Block(n_embd,num_heads) for _ in range(n_layers)] + [nn.LayerNorm(n_embd)])
    self.lm_head=nn.Linear(n_embd,vocab_size)
    self.ln_f = nn.LayerNorm(n_embd)
    self.block_size = block_size

  def forward(self,idx,targets=None):
    B, T = idx.shape[0], idx.shape[1]
    token_embeddings = self.token_embedding_table(idx) #(batch, time, channel) tensor
    position_embeddings = self.position_embedding_table(torch.arange(T,device=device)) #(batch, time, channel) tensor
    x = token_embeddings + position_embeddings
    x = self.blocks(x)
    x = self.ln_f(x)
    logits = self.lm_head(x) #(batch, time, vocab_size) tensor
    #we want (B,C,T) NOT (B,T,C)
    if targets is None:
      loss=None
    else:
      B,T,C=logits.shape
      logits=logits.view(B*T,C) #T represents sequence length (8)
      targets=targets.view(B*T) #C = size of vocab and results in output of 65 logits
      loss = F.cross_entropy(logits,targets) #how well are we predicting?
    
    return logits,loss

  def generate(self,idx,max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
      logits,loss=self(idx_cond) #get predictions, we ignore loss
      logits=logits[:,-1,:]             #focus on last time step
      probs=F.softmax(logits,dim=-1)   #apply softmax 
      idx_next=torch.multinomial(probs,num_samples=1) #sample from distribution
      idx=torch.cat((idx,idx_next),dim=1) #append sampled index to sequence
    return idx




model=BigramLanguageModel(vocab_size)
model.to(device)
logits,loss=model(xb,yb)
print(logits.shape)
print(loss)

idx=torch.zeros((1,1),dtype=torch.long,device=device) #starts off the sequence
print(decode(model.generate(idx,max_new_tokens=100)[0].tolist())) #index to 0th row

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for iter in range(max_iters):
  if iter % eval_interval == 0:
    losses=estimate_loss()
    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
  xb,yb=get_batch('train')
  logits,loss=model(xb,yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

print(f"Final loss: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")



context=torch.zeros((1,1),dtype=torch.long,device=device)
print(decode(model.generate(idx,max_new_tokens=100)[0].tolist())) #index to 0th row