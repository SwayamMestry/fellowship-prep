import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 300
lr = 0.001
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

class Head(nn.Module):
  
  def __init__(self,head_size):
    super().__init__()
    self.key = nn.Linear(n_embed,head_size, bias = False) # key nn taking 32 inputs outputting 16 number keys. (4,8,32) to (4,8,16)
    self.query = nn.Linear(n_embed,head_size, bias = False) # query nn taking 32 inputs outputting 16 number queries. (4,8,32) to (4,8,16)
    self.value = nn.Linear(n_embed,head_size, bias = False) # value nn taking 32 inputs outputting 16 number values. (4,8,32) to (4,8,16)
    self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size))) # lower triangular matrix of ones. 
    '''
    register_buffer is PyTorch's way of saying: "this tensor belongs to the model and should travel with it (device, save/load)
    but it's not a learnable parameter, don't compute gradients for it, don't let the optimizer touch it."
    '''

  def forward(self,x):
    B,T,C = x.shape
    q = self.query(x) # query asks "what this current token is looking for"
    k = self.key(x) # key asks "what imformation is this current token providing"
    v = self.value(x) # value at last asks "what information should this current token relay when being attended to"

    w = q @ k.transpose(-1,-2) * C**-0.5 # transpose(-1,-2) swaps last 2 dimensions so (4,8,16) becomes (4,16,8) which is what we matrix multiply with (4,8,16) to get (4,8,8) which we want (query of each token interacting with every other token's key in the sequence)
    '''
    sqrt specifically since we need to scale down using std dev. imagine 16 coin tosses +1 for heads -1 for tails.
    realistically a score of +/- 16 is extremely rare (think of normal distribution curve)
    usually +4 or -6 somehting like that. similarly instead of divinding by variance we divide by std dev
    which is sqrt of variance. variance here: since we are adding 16 terms together, variance increases 16x
    std dev increases 4x so dividing by std dev gives terms roughly scaled to original terms.
    '''
    w = w.masked_fill(self.tril[:T,:T] == 0,float('-inf')) # fills matrix w with float('-inf') wherever the mask tril==0 is true (so upper triangle excluding diagonal) think of it like the future tokens cannot communicate with the past
    w = F.softmax(w,dim=-1) #softmax converts raw numbers into probabilities by row
    out = w @ v
    return out

class MultiHeadAttention(nn.Module):
  
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) # in place of a regular list. essentially tells pytorch that this is a model part and parameters should be tracked. similar to register_buffer but for lists.
    self.proj = nn.Linear(n_embed,n_embed)

  def forward(self,x):
    out = torch.cat([h(x) for h in self.heads], dim = -1) #concatenates all the outputs of heads along the last dimension
    out = self.proj(out) 
    return out
  
class FeedForward(nn.Module):
  
  def __init__(self, n_embed):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embed, 4*n_embed),
      nn.ReLU(),
      nn.Linear(4*n_embed,n_embed), # projection layer
    )
    
  def forward(self,x):
    return self.net(x)

class Block(nn.Module):
  
  def __init__(self, n_embed, num_heads):
    super().__init__()
    head_size = n_embed//num_heads # divides 32 by number of heads so when they are concatenated they become back to 32 regardless of number of heads
    self.sa = MultiHeadAttention(num_heads, head_size)
    self.ffwd = FeedForward(n_embed)
    self.ln1 = nn.LayerNorm(n_embed)
    self.ln2 = nn.LayerNorm(n_embed)

  def forward(self, x):
    x = x + self.sa(self.ln1(x)) # added a residual learning layer that passes gradient directly to input (add function passes gradients directly) since the model is too deep and suffers from vanishing gradients.
    x = x + self.ffwd(self.ln2(x))
    return x

class BigramLM(nn.Module):

  def __init__(self):
    super().__init__() # essentially to inherit the __init__() of the superclass (nn.Module)
    self.token_embedding_table = nn.Embedding(vocab_size,n_embed) # embedding table with 65 rows each with 32 values random numbers instead of 65
    self.positional_embedding_table = nn.Embedding(block_size,n_embed)
    self.blocks = nn.Sequential(
      Block(n_embed,4),
      Block(n_embed,4),
      Block(n_embed,4),
      nn.LayerNorm(n_embed)
    )
    self.lm_head = nn.Linear(n_embed,vocab_size) # neural network layer with 32 inputs per layer and 65 such neurons leading to 65 outputs like before. no activation.

  def forward(self, i, targets=None):
    B,T = i.shape
    tok_emb = self.token_embedding_table(i) # (Batch, Time, Channel) takes a row of 32 numbers for each i
    pos_emb = self.positional_embedding_table(torch.arange(T)) #(T,C)
    x = tok_emb + pos_emb #(B,T,C)
    x = self.blocks(x)
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
      i_cond = i[:,-block_size:]
      logits, loss = self(i_cond) #runs forward() on each i in the (4,8) xb
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