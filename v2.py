import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
lr = 0.01
eval_iters = 200
n_embed = 32

torch.manual_seed(1337)

dataset = open('input.txt', 'r', encoding='utf-8')
text = dataset.read()

chars = sorted(list(set(text)))  #the first element is actually '\n' and not ' ' like andrej said
vocab_size = len(chars)

encode_map = {ch:i for i,ch in enumerate(chars)} #maps characters to integers
decode_map = {i:ch for i,ch in enumerate(chars)} #maps integers back to characters
encode = lambda s: [encode_map[c] for c in s] #assigns each character of the string 's' to an int from the map 'encode_map'
decode = lambda l: ''.join(decode_map[c] for c in l) #gets the corresponding character for every integer in the list created by the 'encode_map' for the string

data = torch.tensor(encode(text), dtype=torch.long) # coverting all of the input.txt into encoded list of integers for each character. torch.long makes sure it is stored as integers.
n = int(0.9* len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split_type):
  data = train_data if split_type == 'train' else val_data
  ix = torch.randint(len(data)-block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  return x,y

@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train','val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      x,y = get_batch(split)
      logits,loss = model(x,y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out

class BigramLM(nn.Module):

  def __init__(self):
    super().__init__() # essentially to inherit the __init__() of the superclass (nn.Module)
    self.token_embedding_table = nn.Embedding(vocab_size,n_embed) # embedding table with 65 rows each with 32 values random numbers instead of 65
    self.positional_embedding_table = nn.Embedding(block_size,n_embed)
    self.lm_head = nn.Linear(n_embed,vocab_size) # neural network layer with 32 inputs per layer and 65 such neurons leading to 65 outputs like before. no activation.

  def forward(self, i, targets=None):
    B,T = i.shape
    tok_emb = self.token_embedding_table(i) # (Batch, Time, Channel) takes a row of 32 numbers for each i
    pos_emb = self.positional_embedding_table(torch.arange(T)) #(T,C)
    x = tok_emb + pos_emb #(B,T,C)
    logits = self.lm_head(x) # (B,T,vocab_size) feeds the 32 numbers as input to a linear NN layer with 65 neurons. each neuron has 32 weights, 1 for each input and 1 bias. total 65 outputs 1 per neuron.

    if targets == None:
      loss = None
    else:
      B,T,C= logits.shape
      logits = logits.view(B*T,C)#3D to 2D compresses (4,8,65) to (32,65)
      targets = targets.view(B*T) #2D to 1D compresses (4,8) to (32,)
      loss = F.cross_entropy(logits,targets)
    return logits , loss

  def generate(self,i,max_new_tokens):
    for _ in range(max_new_tokens):
      logits, loss = self(i) #runs forward() on each i in the (4,8) xb
      logits = logits[:,-1,:] #takes only the last character of each of the 4 rows so shape is now (4,65)
      probs = F.softmax(logits,dim=-1) #converts logits to probabilities
      i_next = torch.multinomial(probs, num_samples = 1) #randomly picks one number (from the 65 available) weighted by the probabilities
      #i_next = torch.argmax(probs, dim=-1, keepdim=True)  # instead of torch.multinomial(probs, num_samples=1)
      i = torch.cat((i,i_next), dim=1) # concatenates the 4 new i_next (shape is (4,1)) to the existing xb (shape of xb goes from 4,8 to 4,9 and so on every iteration)
    return i
  
model = BigramLM()
optimizer = torch.optim.AdamW(model.parameters(),lr=lr)

for iter in range(max_iters):
  if iter%eval_interval==0:
    losses = estimate_loss()
    print(f'step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}')
  xb,yb = get_batch('train')
  logits,loss = model(xb,yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

context = torch.zeros((1,1),dtype = torch.long)
print(decode(model.generate(context,max_new_tokens=500)[0].tolist()))