# AI Fellowship Prep

Public log of learning ML/RL for the Anthropic Fellowship application.

## Day 1 (July 1)
Watched the intro and conceptual setup for backprop. No code yet, just building intuition for what a derivative actually tells you (how sensitive an output is to a tiny nudge in an input) and how the chain rule combines that together across a computation graph.

## Day 2 (July 2)
Built the 'Value' class with '_children', '_op', '_backward', and '_prev'. Worked through manual backprop on a full expression by hand and hit a real bug myself (calling '.backward()' twice without resetting grads) before Karpathy even covered it in the video helped a lot to debug it first, then see the same fix explained haha.

**Things I found today:**
- What a "local derivative" actually is: the derivative of a parent op w.r.t. its child, multiplied by the parent's '.grad' to get the child's gradient contribution
- Why grads accumulate ('+=') instead of overwrite, same mechanism PyTorch uses for gradient accumulation across mini-batches when a full batch won't fit in memory
- Why '_children' defaults to an empty tuple (leaf nodes like 'Value(2.0)' have no parents) but gets converted to a 'set' internally (avoids double-processing a node that's used twice in the same operation, e.g. 'a*a')
- Confirmed: node-used-twice-in-one-op (handled by the set) and node-used-in-two-separate-ops (handled by '+=' in each op's '_backward') are different mechanisms that both end up mattering for correct gradients

## Day 3 (July 3)
Neuron backprop example done, then implemented '_backward' for '__add__' and '__mul__' MYSELF, felt pretty good about that. Also did tanh's '_backward' and got '.backward()' working on a full expression graph with the topological sort.

**Things that confused me but got sorted out:**
- Was confused why tanh's children is '(self,)' and not '(self, other)' turns out tanh only has ONE input, and that comma is doing real work (makes it a tuple, not just 'self' in parentheses)
- 'tanh(x)' derivative is '1 - tanh^2(x)' need to just remember this one
- '+' and '-' just pass gradient straight through (local derivative = 1), '-' flips the sign, that's it

Productive day overall.

## Day 4 (July 5)

Finished the ENTIRE 'Value' class today '__neg__', '__sub__', '__pow__', '__rpow__', '__truediv__', '__rtruediv__', all derived and debugged myself, no answers given, just pointers. Fixed my own bugs along the way: '__pow__' briefly treating 'other' as a Value, '__truediv__' missing the 'out._backward =_backward' line, '__rtruediv__' causing infinite recursion ('return other/self' calling itself), exp's '_backward' using '=' instead of '+=', '__rsub__' calling '__neg__(self)' like a function instead of using '-self'. Then built 'Neuron', 'Layer', 'MLP' and explained the 'size = [nin] + nouts' layer-chaining trick back in my own words.

**Things I understood today:**
- Why mul's formula ('other.data * out.grad') and exp's formula ('out.data * out.grad') aren't the same thing every operation computes its own local derivative from its own math, some just happen to reuse 'out.data' as a shortcut (exp and tanh), others need a different value entirely (mul needs 'other.data')
- Derived x^n's derivative myself: 'n * x^(n-1)'
- Why 'x^y' (both Values) needs a second, different formula for the exponent (log-based: 'x^n * ln(x)'), and why real implementations restrict 'other' to a constant instead (x>0 restriction on log, added complexity for a rare case)
- '__rpow__' is genuinely needed (unlike '__rmul__'), and worked out its correct formula and parameter roles myself
- The recurring gotcha across '__rtruediv__' and '__rsub__': 'self' is always the object whose method Python actually called, not "whichever operand is written first" this is what caused my infinite recursion bug in '__rtruediv__'
- 'requires_grad=True'/'False' and 'torch.no_grad()' track gradients only for what you'll update (weights), not raw input data, for memory/compute efficiency plus the interpretability exception where you flip that on the input on purpose
- '.data' vs '.item()' on tensors, and that '.item()' alone (no '.data' needed) is the modern way
- Chased a real ~10^-7 gradient mismatch vs PyTorch (tested float32 vs float64, 'torch.Tensor' vs 'torch.tensor', manual tanh vs 'torch.tanh') genuinely unresolved, left it since it doesn't affect correctness
- 'Layer': 'nin' = inputs per neuron, 'nout' = neuron count, MLP's 'size = [nin] + nouts' auto-chains each layer's output size to the next layer's input size

**Notes:**
- '__truediv__'/'__rtruediv__': noted these also work as a one-liner ('self*other^-1') since mul/pow are already defined, but wrote the full explicit version to actually understand the derivation
- On 'o.item()': "apparently modern pytorch does not need .data like andrej did"

## July 6
Finished the whole micrograd video. Trained the MLP to convergence on the toy dataset final predictions matched targets almost exactly (e.g. ~0.998 vs 1.0). Tested the learning rate hands-on check: 0.01 and 0.1 barely differed since tanh's bounded gradients make this tiny network unusually robust. At lr=1.0 on a freshly initialized network, the loss got stuck oscillating around 8.0 for ~13 iterations (overshooting the minimum each step) before by chance landing close enough to escape into normal smooth convergence. Real overshoot behavior, just not the instant-blowup I expected going in.
Micrograd is done full Value engine (every op derived and debugged myself), Neuron/Layer/MLP, working training loop. Next: nanoGPT.

**Doubts I had today, sorted out:**
- 'sum()' threw a TypeError. turned out sum()'s hidden default 'start=0' is a plain int, so 'Value' needed a missing '__radd__' (same pattern as rmul/rpow/rtruediv/rsub, just for '+')
- Confirmed 'params += p' and 'params.extend(p)' are functionally identical for lists (list's '__iadd__' mutates in place), but 'params = params + p' is genuinely different (creates a new list)
- Got the full breakdown of how the nested list comprehension '[p for neuron in self.neurons for p in neuron.parameters()]' maps to my original explicit loop
- Confirmed 'p.data -= 0.01 * p.grad' and 'p.data += -0.01 * p.grad' are mathematically identical, pure style difference
- Learning rate 0.1 had no visible effect at first traced to tanh's bounded gradient (max derivative of 1, shrinking fast away from 0), making this tiny network unusually resistant to instability
- Rebuilding my own MLP from memory, found two real bugs myself: 'self.b' wasn't wrapped in 'Value(...)' (broke '.tanh()'), and 'MLP.__init__' looped 'range(len(size))' instead of 'range(len(nouts))'

## July 7
Started nanoGPT. Built the full data pipeline (tokenizer, encode/decode, train/val split, get_batch) and the BigramLM class embedding table, forward() with cross_entropy loss, generate() with the sampling loop. Got the first loss reading on the untrained model (4.8786).
Tested argmax vs multinomial for generation, correctly predicted it'd get stuck in a repeating loop before running it, then traced the "identical output every run" result back to the actual cause: the seed only controls the embedding table's random init, since argmax and my fixed starting idx introduce zero randomness of their own.
Stopped right before training the bigram model self-attention is next.

**Doubts I had today, sorted out:**
- Why store the dataset as a tensor instead of a plain list. speed, GPU compatibility, and needing 'torch.long' specifically since embedding lookups require integer indices, not floats
- Why CUDA and GPUs actually are. thousands of parallel cores vs. a CPU's mostly-sequential execution
- Why feeding the whole text into the transformer at once is expensive. attention cost scales with block_size^2, confirmed by hand (doubling block_size is 4x cost)
- Full breakdown of get_batch's 'ix'/'x'/'y' lines, especially what '(batch_size,)' means as a shape tuple, and correctly worked out '(4,)' vs '(4,1)' vs why '(,4)' isn't valid syntax
- 'super().__init__()' and subclassing 'nn.Module' worried I couldn't build this kind of logic myself, but understood the inheritance in it
- Full 'generate()' walkthrough using my actual xb values, including why it only looks at 'logits[:,-1,:]' (bigram = 1-character memory) and why block_size=8 exists anyway (scaffolding for attention, not needed yet)

## July 9
Trained the bigram model using the AdamW optimizer and a proper training loop, added estimate_loss() to track train/val loss averaged over many batches instead of a single noisy one. Fixed a real bug in decode() it was wrapping the output in an extra list, so print() showed literal $'\n'$ instead of real line breaks.
Slower day overall, didn't make it past the script section into version 1 of self-attention. Picking up there next time.
**Doubts I had today, sorted out:**
- Why estimate_loss averages over 200 batches instead of just checking one. reducing variance/noise in the loss estimate, since one random batch could get lucky/unlucky by chance
- What `@torch.no_grad()` actually does and why it's used as a decorator worked out that it's shorthand for wrapping the whole function body in 'with torch.no_grad():', and that a decorator must take a function in and return a function out
- Tested my own understanding by writing a decorator from scratch (`@sub(a,b)` on 'add') correctly identified why my first version wouldn't actually work (sub didn't take a function as input or return one), then fixed it and correctly traced through the corrected version by hand

## July 10
Version 1 of self-attention (averaging past context with for loops), version 2 (the matrix multiply trick replacing the double for-loop), version 3 (adding softmax with -inf masking so future positions get exactly zero weight instead of leaking a small nonzero probability). Understood why 0 doesn't work for masking but -inf does. traced through why the last row can look "correct" by coincidence even when the masking logic is actually broken everywhere.
Karpathy created a new v2.py restructuring BigramLM. embedding table now goes vocab_size to n_embed (32) instead of directly to vocab_size, with a new lm_head (nn.Linear) projecting back up to vocab_size, plus positional embeddings added. Mapped this directly onto my own Neuron/Layer code from micrograd. confirmed nn.Linear has no activation function built in at all, and identified that positional embeddings are currently inert scaffolding, since nothing yet actually uses position-specific information.
Stopped right before "the crux of the video" actual self-attention (version 4) is next.

**Doubts I had today, sorted out:**
- Randomly encountered bag-of-words vectorization in xbow's naming.
- Why softmax with 0s instead of -inf doesn't correctly mask future positions, and why the very last row can look right anyway (uniform inputs, not because the masking is actually correct there)
- B,T,C shapes and the mean_{i<=t} notation, connected the math notation directly to the nested for-loop mechanically
- Why tok_emb now goes to n_embed instead of vocab_size directly, and what lm_head (nn.Linear) does. mapped it onto my own Neuron/Layer dot-product logic
- Confirmed nn.Linear has NO activation function built in. has to be added as a separate step, same as my own Neuron's two-line structure (weighted sum, then .tanh())
- Positional embeddings. confirmed they're currently doing nothing meaningful without attention yet, just inert scaffolding for now

## July 11
Version 4: the actual self-attention mechanism. Built Q/K/V from scratch (with hints), understood why bias=False for all three (dot-product comparison doesn't benefit from a fixed offset), why scaling by sqrt(head_size) specifically. worked through the coin-flip variance/std-dev reasoning until it actually made sense.
Encapsulated everything into a Head(nn.Module) class, learned register_buffer (fixed, non-learnable tensor that still travels with the model for device/save-load, unlike a plain attribute or a learnable parameter). Inserted a single self-attention head into BigramLM between the embeddings and lm_head. Added block_size cropping (i_cond = i[:,-block_size:]) in generate() — needed now that attention looks at the whole sequence, not just the last character like bigram did.
Asked (and parked) a deeper question about what it even means for q/k/v to "know" anything before training. settled that meaning emerges from gradient descent, not design, and the actual "seeing" of learned patterns is Week 2's TransformerLens job, not something to chase now.
Stopped right before multi-headed self-attention.

**Doubts I had today, sorted out:**
- Why q/k/v are meaningless as random numbers before training, and how meaning actually emerges. traced back to the same mechanism as the bigram embedding table becoming meaningful only through training
- What register_buffer does and why tril needs it instead of being a plain attribute
- What self.tril[:T,:T] slicing does. sized for the max block_size, sliced down to the actual current sequence length
- Why i_cond = i[:,-block_size:] is needed in generate() now, when it wasn't for bigram. attention needs all positions, and the model was only trained on block_size-length chunks anyway
- Why bias=False for key/query/value. the comparison-based role doesn't benefit from a fixed offset the way lm_head's does

## July 12
Built multi-head attention (MultiHeadAttention with proj to mix across heads. correctly reasoned through why per-head feedforwards would be redundant with a single combined one, and why proj is genuinely needed since concatenation alone doesn't mix anything). FeedForward with the proper 4x expansion (n_embed to 4*n_embed back to n_embed). Understood residual connections properly. traced why deep networks need them (vanishing gradients through many stacked layers) and why blocks "come online slowly" during training (network initially relies on the x passthrough since f(x) starts near-random).
Went deep on batch norm vs layer norm: worked through gamma/beta as a learned escape hatch from forced normalization, running_mean/running_var as the fix for meaningless single-example statistics at inference, and momentum as noise-smoothing across batches. using Karpathy's actual BatchNorm1d class from makemore as the reference. Traced through pre-norm vs post-norm precisely. LayerNorm sitting in the residual's main path (post-norm) vs. only in the side-branch feeding into sublayers (pre-norm), and why that matters for keeping the gradient "highway" clean across many stacked blocks.
Stopped right before scaling up the model.

**Doubts I had today, sorted out:**
- Why matrix multiplication and dot products aren't different things. a matmul result is just many dot products computed at once, traced through with real small numbers before connecting it to the actual 8×16 shapes in the code
- Whether per-head feedforwards before concatenation would help (they'd be redundant with a single combined feedforward, which already has access to everything) vs. whether proj is needed (yes, concatenation alone doesn't mix information, proj is the first real mixing step)
- Full mechanics of batch norm (gamma/beta, running stats, momentum) using Karpathy's actual class, then extended to why layer norm skips the running-stats/train-eval-mode complexity entirely
- 2D and 3D worked examples of batch norm vs layer norm. which axis gets normalized in each case
- Why pre-norm (not post-norm) keeps gradients flowing cleanly through residual connections. LayerNorm sits in the side-branch, not the main path

## July 13
Reviewed multi-head attention/feedforward/block code from yesterday, then moved into dropout (three separate placements. inside each Head after softmax, after MultiHeadAttention's proj, after FeedForward's output. 48 total dropout applications across 6 layers * 8 per block). Scaled the model up to real config (block_size 256, n_embed 384, 6 heads, 6 layers). Set up MPS on Mac, hit a device-mismatch bug, fixed it, then set up CUDA on Colab and compared MPS (~300 steps/10+ min, ran hot) vs CUDA T4 (~300 steps/5 min) killed the slower Mac run once the comparison was clear.
Finished the entire video: encoder vs decoder notes, full nanoGPT walkthrough, ChatGPT/GPT-3/pretraining vs finetuning/RLHF context, conclusions. Trained the full scaled-up model on Colab T4 ~1hr 15min for 5000 steps, loss 4.28 to 1.09 (train)/1.48 (val). Generated real Shakespeare-structured text. correct character/dialogue formatting, real words, grammatically plausible fragments, not semantically coherent (expected at this scale).

**02_nanogpt.ipynb is DONE. WEEK 1 COMPLETE.**

**Doubts I had today, sorted out:**
- Whether dropout was one layer applied broadly. corrected three genuinely separate dropout layers, each guarding a different stage (attention weights, combined multi-head output, feedforward output), never touching the residual path itself
- Caught my own bug. passed the global 'n_heads' instead of the 'num_heads' parameter in 'MultiHeadAttention.__init__
- Real device debugging: 'torch.arange(T)' defaulted to CPU while the rest of the model was on MPS, causing a runtime error. fixed with explicit device=device
- Empirically compared MPS vs CUDA vs CPU speed instead of just reading about it. Colab's free T4 clearly won
- What GELU is and how it differs from ReLU. a smoothed curve instead of a sharp corner at zero
- The "dying ReLU" problem. neurons stuck in the always-negative zone get exactly zero gradient forever, since ReLU's derivative there is 0
- Why GELU fixes this. no truly flat, zero-derivative region, so gradient always flows at least a little
- Leaky ReLU as an alternative fix (small nonzero slope for negative inputs) but it's an arbitrary hyperparameter with mixed real-world results, and still has a sharp kink unlike GELU's smooth curve

## July 15
Read Attention Is All You Need through section 3.5 (Positional Encoding). Found three differences from Karpathy's implementation:
(1) the paper uses post-norm (Add & Norm after the sublayer) vs Karpathy's pre-norm. the paired task answer.
(2) the paper ties the embedding table and lm_head to the same weight matrix (weight tying), Karpathy's version uses two fully separate ones.
(3) the paper uses fixed sine/cosine positional encoding, Karpathy uses a learned embedding table. both perform similarly.
Confirmed the Q/K/V diagram (Figure 2) maps exactly onto my own Head.forward() and MultiHeadAttention code, box for box. Connected "output embeddings offset by one position" directly to my own y = x shifted by one in get_batch. built that mechanism before knowing the paper's name for it.

**Doubts I had today, sorted out:**
- Weight tying: initially thought it meant two separate tensors with the same shape, actually means literally the same underlying tensor used in two directions (one assignment, no copy)
- Where the "transpose" in weight tying actually happens. not something I write, it's baked into every nn.Linear's forward pass (x @ weight.T), and PyTorch stores Linear weights as (out_features, in_features), which is why the embedding table and lm_head's weight shapes match despite looking reversed
- Sine/cosine positional encoding at a high level (different-frequency waves per dimension create a unique fingerprint per position) didn't go deep into the trig, noted it's not critical right now.

## July 16
Continued reading Attention Is All You Need past 3.5, through Section 4 (Why Self-Attention) and Section 5 (Training Data/Batching, Hardware, Optimizer, Regularization). Worked through Table 1 (complexity/sequential-ops/max-path-length for self-attention, recurrent, convolutional, restricted self-attention) until the n<d reasoning for why self-attention beats recurrent layers actually clicked (worked concrete numbers, n=50 d=512, not just the variables). Untangled restricted self-attention's sliding local window from my own block_size causal cutoff, different mechanisms entirely. Worked through BPE/word-piece tokenization and the paper's LR warmup+inverse-sqrt-decay formula, including how it'd actually get implemented in PyTorch.
Found three more differences from my own build, on top of the three from July 15:
(4) the paper uses a warmup (4000 steps) + inverse-square-root decay learning rate schedule, I used a flat constant LR for the whole run.
(5) the paper applies dropout to each sub-layer's output before the residual add plus to the embedding+positional sum, I applied dropout to attention weights (post-softmax), post-projection, and post-feedforward instead.
(6) the paper trains with label smoothing (e=0.1, soft targets), I used plain hard-target cross_entropy, no smoothing.

**Doubts I had today, sorted out:**
- Why self-attention (O(n^2 d)) beats recurrent (O(n d^2)) specifically when n < d, worked concrete numbers instead of just the variables
- Restricted self-attention's sliding window vs my own block_size. sliding window moves with every token and never fully cuts off distant context, block_size is a hard wall, disjoint chunks with zero visibility across the boundary
- Whether a sliding window could look both directions vs causal-only. paper's original formulation is centered/bidirectional, but that only works for encoder-style models like BERT, not autoregressive ones like what I built
- What BERT actually does, since I didn't really know. masked language modeling, bidirectional context, produces representations for downstream tasks, doesn't generate text like GPT does at all
- How BPE builds its vocabulary from raw characters via repeated pairwise merging, same idea as my own encode/decode, just with a merging step on top
- The LR schedule formula (warmup then inverse-sqrt decay, peaks exactly at step=warmup_steps) and how it'd actually be implemented in PyTorch
- Section 5.4 actually describes THREE regularization applications under two headers, not two. residual dropout covers two separate placements (sub-layer output, embedding sum) plus label smoothing separately

## July 17
Finished the rest of the paper. Section 6 (results) through conclusion plus the attention visualization figures. PAPER FULLY READ, end to end.
Worked through beam search from scratch with a full step-by-step example (vocab of 6 tokens, beam size 2), understanding it as running my own generate() loop on 4 sequences in parallel instead of 1, with a global pool-and-keep-top-k step after every token that has no equivalent in what I built. Also worked through checkpoint averaging and the length penalty formula (a division/normalization, not subtraction). Two more differences found:
(7) the paper uses beam search (beam=4) with a length penalty at inference, I only ever used greedy (argmax) or multinomial sampling.
(8) the paper averages the last 5 (base)/20 (big) checkpoints into the final model, I used the single final checkpoint as-is.

Went through Table 3 (Model Variations, full ablation study) row by row: attention head count vs head size tradeoff (fixed total budget, can't maximize both without growing d_model), d_k independent from d_v (only d_q and d_k must match, hard requirement from the dot product), more layers isn't strictly better (N=8 actually scored worse BLEU than base's N=6 despite lower perplexity and more params), confirmed learned positional embeddings score basically identical to sinusoidal.
Read 6.3 (constituency parsing), a generalization test on a completely different task. Got into the actual mechanics of semi-supervised learning here: WSJ-only is real human-labeled data, the BerkeleyParser/high-confidence corpora are auto-labeled by existing parsers with zero human involvement. Worked through why the semi-supervised run scored higher (~400x more data, imperfect labels average out at scale), and the genuinely interesting result that the trained model matched/beat the very parser that generated its own training labels.
Finished on the attention visualization figures (long-distance dependency tracking, anaphora resolution, heads specializing in different jobs). Plan: build the same thing myself by saving attention weights out of my own Head.forward() instead of discarding them, then compare against TransformerLens once I get there in Week 2.

**Doubts I had today, sorted out:**
- Beam search's "keep only the global top-k across all branches" step. worked a full 3-step example by hand to see a sequence that started behind get overtaken by a better continuation
- Why beam search isn't exponentially expensive despite exploring every next-token option per beam. cost stays a constant multiple of greedy, specifically because of the discard-to-top-k step every round
- Length penalty formula. log P(Y) divided by a length-based normalization term, not subtraction, alpha is a tunable exponent (0 = no correction, 1 = full linear correction)
- Table 3 row A. not "n_embed needs to be large enough", d_model is held fixed the whole row, num_heads and head_size are two ways of splitting one fixed budget. confirmed separately via the "big" model that growing d_model does let you support more heads without shrinking head_size
- Table 3 row B. d_q and d_v don't need to match, only d_q and d_k do, the paper's own d_k=16/d_v=64 ablation proves this directly
- Table 3 row C. more layers isn't automatically better. N=8 scored worse BLEU than base's N=6 despite lower perplexity and more params
- Table 3 row E. testing sinusoidal vs the exact learned-embedding-table approach I actually built, nearly identical scores, paper picked sinusoidal only for extrapolation to unseen sequence lengths
- Semi-supervised vs self-supervised, since I conflated them. semi-supervised here means small human-labeled data plus a much larger machine-auto-labeled corpus, unrelated to GPT's self-supervised next-token training
- What the BerkeleyParser/high-confidence corpora actually are. an existing separate parser auto-labeling millions of raw unlabeled web sentences, high-confidence requires two independent parsers to agree before keeping a sentence
- Why a model trained on an imperfect teacher's labels can end up beating that teacher. the teacher's mistakes are inconsistent noise that doesn't reinforce across millions of examples, real patterns do

## July 18
Read The Illustrated Transformer (Jay Alammar) front to back. No new code, but extensive digging into my own build's mechanics, prompted by mapping the blog's diagrams onto Head/MultiHeadAttention/Block. Confirmed the "independent per position" language in the post means no cross-position dependency, not literal looping. self.ffwd(x) is one single vectorized call over the whole (B,T,C) tensor, not called separately per position.
Worked through why the last position (T-1) has full context of everything before it, tracing directly through my own tril masking mechanism with a concrete 4-word example. Extended into a deeper question: what if there was no masking and every position could see every other? Worked out the leakage problem directly (the model would see the literal answer it's predicting), and why that's exactly why BERT can only do masked-word-fill, not next-token prediction, since it has no causal mask.
Tested a good instinct that turned out wrong: tried averaging/combining all T positions' logits for generation instead of just the last one. Worked out why this doesn't help: every position is trained on a different shift-by-one target, not the same question with more or less context, and the last position already has full context via causal attention, so earlier positions don't add missing information, they just answer irrelevant questions.
Full recap of terminology (batch, block_size, B/T/C, head, ffwd, dropout) and the exact order of operations through one Block, plus what lm_head and the train/generate split actually do at the end of the pipeline. Multi-head shapes walked through in detail with real numbers: same input matrix fed to every head (different weight matrices per head), concat happens along the feature axis per position (not across positions), proj is the real mixing step.
Went deep on cross-entropy/KL divergence with worked numeric examples, both for hard targets (H(p)=0, H(p,q)=KL(p,q) exactly) and with label smoothing applied (H(p) becomes nonzero, H(p,q) and KL diverge by exactly H(p)). confirms and quantifies the paper's claim from a few days ago that label smoothing "hurts perplexity."

**Doubts I had today, sorted out:**
- Why "independent per position" for FFwd doesn't mean literally looped/called separately, it's one vectorized matmul, nn.Linear applies to the last dim and broadcasts over B and T automatically
- Why position T-1 has context of everything before it, direct consequence of tril's triangular structure, nothing special about that position except it's at the end
- Whether removing causal masking would let every position have equal info for free. yes for raw access, but it would leak the actual next-token answer directly into the computation used to predict it, breaking the entire training task
- Why averaging/combining all positions' logits for generation isn't better than using just the last one, every position has a different training target (shift-by-one), last position already has full context via attention, earlier ones don't add anything useful
- Found the exact shift-by-one line in get_batch: y = torch.stack([data[i+1:i+block_size+1] for i in ix])
- Whether batch elements need to be contiguous, yes within a single sequence always, though different sequences in the batch can start anywhere/overlap freely
- Confirmed wei stays (B,T,T) after softmax, softmax doesn't collapse a dimension, just rescales values within it
- Positional encoding numbers in the blog's toy example are fixed (sin/cos formula), not random and not trained, unlike my own learned position_embedding_table
- Cross-entropy vs KL divergence, worked with real numbers for both hard-target and label-smoothed cases to see exactly where and why they diverge

No commits to model code today, pure reading/Q&A session.

## August 5
Ran a full-journey recap spanning micrograd through the Illustrated Transformer, 15 questions instead of the usual 8. Landed gaps on beam search (had the wrong pruning mechanism, it's a fixed top-k kept at every step, not dropping the two lowest-scored) and the length penalty (had the direction backwards, it corrects for a bias toward short sequences, doesn't punish long ones). Also mixed up "attention weights" with weight matrices in the dropout-locations answer, and missed weight tying and the LR warmup+decay schedule as two of the 8 differences from the paper.
Went back into 02_nanogpt.ipynb and added block-level intro comments above Head, MultiHeadAttention, FeedForward, Block, BigramLM, and the training loop. First attempt had a bug the comment blocks were unindented, sitting outside the class bodies instead of as the first statement inside them, which throws an IndentationError. Fixed.
Finished the rest of the 3Blue1Brown series. Attention's O(n^2) bottleneck (the $q@k$ matrix is T*T, quadratic in context length) and the research directions around it. sparse attention and Longformer restrict which positions get compared, Linformer shrinks K/V into a smaller fixed dimension, Reformer hashes instead of comparing everything, adaptive attention span lets each head learn its own context length, ring attention distributes the same O(n^2) cost across multiple GPUs instead of reducing it. Also the MLP-as-fact-storage framing. up-projection rows act as learned concept detectors, ReLU gates which ones fire, down-projection rows inject the associated fact back into the residual stream.
Revisited cross-entropy from the Shannon/compression side instead of just the loss-function side. Worked an example. 4 symbols at 50/25/12.5/12.5% probability, built actual valid prefix-free binary codes for them, and the resulting code lengths (1, 2, 3, 3 bits) matched -log2(p) for each symbol exactly, showing why -log(p) was the natural choice for optimal coding before it ever got reused as a loss function.
Added LeetCode to the plan and solved the first 3, directly on LeetCode. Two Sum (hashmap, one pass). caught a bug where I was building a tuple key instead of two separate lookups. Valid Parentheses (stack). went through several bugs in sequence, returning immediately on the first successful match instead of letting the loop continue, trying to use a ternary where 'continue' doesn't belong (a ternary can only choose between two return values, not conditionally skip a 'return' entirely), an uninitialized stack, and no final check for leftover unmatched brackets. then refactored the three repetitive if/elif branches into one dict lookup. Best Time to Buy and Sell Stock (single pass, running min + running profit). first instinct was checking against a single fixed global minimum, worked out with a concrete counterexample [3,2,6,5,0,3] why the minimum has to update as you go instead.

**Doubts I had today, sorted out:**
- Beam search's actual pruning mechanism, fixed top-k kept at every step, not dropping the worst few
- Length penalty's actual direction, corrects for a bias toward short sequences rather than punishing long ones
- Why 'is' shouldn't be used to compare strings in Python, even though it can accidentally work for short strings
- Why a ternary can only choose between two return values, it can't conditionally skip a 'return' statement entirely
- Why a running minimum, not a single fixed global minimum, is required for Best Time to Buy and Sell Stock
- MLP up/down projection as a key-value memory, up-projection as concept detectors, ReLU as the gate, down-projection as the values injected into the residual stream
- Why -log(p) was the original optimal code length in Shannon's compression theory, worked with real binary codes matching the formula exactly

## August 6
Wrote 'transformers-explainer.md' (Day 18). walked the full pipeline (tokenization > embedding > attention > feedforward > output) grounded in my own build's actual numbers (block_size 256, n_embed 384, 6 heads at head_size 64), not just the general architecture. First draft had an error. claimed FeedForward mixes information across heads, when that's actually the 'proj' layer's job. fixed. Second draft had another error in the output stage. said FeedForward converts to logits, when that's actually lm_head, a separate linear layer that only runs once at the very end, distinct from any block's FeedForward. Fixed both, explainer done and grounded correctly now.
Started today's session with an 8-question recap covering yesterday's cross-entropy/Shannon material plus the LeetCode bugs from yesterday, plus today's explainer fixes. 7/8 correct. error on Two Sum's brute force complexity, said O(n) when it's actually O(n^2) (confused the per-element check cost with the total cost across all elements).
Solved 3 LeetCode problems today. Group Anagrams (hashmap keyed on sorted string). caught a bug where seeding the output list with an extra empty list at index 0 left a stray empty group in the final output. Reverse Linked List, both iterative and recursive. Iterative went through several bugs in sequence. an early version created a cycle by never updating the original head's own '.next' pointer, then a crash from advancing a pointer past 'None' without checking, fixed with a 'break' guard, then another crash in the setup code for single-node/empty lists, fixed with an upfront edge case check. Recursive had a mixed-up head.next.next = temp (should've been = head) that created self-loop cycles, plus a missing 'return temp' at the end. Valid Palindrome, also both versions. Iterative: used '.isalpha()' instead of '.isalnum()' (was incorrectly skipping digits), and a '!=' loop condition that let the two pointers cross without stopping, causing an out-of-bounds crash on short inputs. Recursive: inverted base case logic, recursive calls missing 'return' entirely, off-by-one slice bounds in all three branches, and after all of that was fixed hit a Memory Limit Exceeded on large inputs because slicing creates a new string copy at every call (~O(n^2) total memory across the recursion). Fixed by passing 'temp1'/'temp2' as index parameters instead of slicing, recursing on the same original string throughout.

**Doubts I had today, sorted out:**
- The actual difference between 'proj' (mixes heads) and 'FeedForward' (doesn't, runs per-position independently)
- The actual difference between 'lm_head' (final logits, once) and 'FeedForward' (per-block, many times)
- Two Sum's real brute force complexity, O(n^2) not O(n) a per-check cost isn't the same as total algorithm cost
- Why recursive Reverse Linked List needs 'head.next.next = head', not '= temp' which pointer is the tail of the reversed sublist vs. which one is the overall new head
- Why a '!=' loop condition can let two inward-moving pointers cross without ever triggering, vs. '<=' catching both the meet and the cross
- Python default arguments can't reference other parameters ('temp2=len(s)-1' fails) since defaults evaluate at function definition time, not call time. use None as a sentinel and compute inside the function body instead
- Why slicing-based recursion blows up memory on large inputs (new string copy per call) while index-based recursion on the same string doesn't

## August 7
Installed transformer_lens, loaded GPT-2 small via HookedTransformer.from_pretrained. Ran three sentences ("hello world!", "a cat and a dog", "a dog and a cat") through layer 0, pulled the attention pattern for all 12 heads each time, and classified every head by what its rows/columns actually did. Hit a bug early on. cache["pattern", layer] keeps the batch dimension, so my first loop was iterating over batch instead of heads and printing whole 6x6 matrices instead of single rows. Fixed by indexing [0] to drop the batch dim.
Findings, held across all three sentences: heads 0, 9, 11 are clean, consistent sink heads (attend to endofsentence every row). Heads 1, 3, 5 are clean, consistent self-attending heads. Heads 2 and 10 lean sink but noisier, not as absolute. Head 8 looked sink-like on "hello world!" but flipped to self-attending mid-sentence on the other two inputs, doesn't hold as one category. Head 7 shows a real previous-token spike late in each sentence. Head 6 turned out to be the cleanest new result. swapped which noun sat in a fixed sentence slot ("a cat and a dog" to "a dog and a cat") and head 6's attention to that slot barely moved (0.29 to 0.34 range regardless of the word there), it's tracking position, not word identity. Head 4 never resolved. some query positions look positional like head 6, one row spiked specifically on "cat" and dropped when "cat" left that slot, so it's doing different things at different positions. Left it open rather than forcing a label.
Added self.attn_weights = w.detach() to my own Head.forward() (in v2.py) to cache the attention pattern instead of discarding it. Learned the actual mechanism behind why this works without touching Head's return signature or MultiHeadAttention's forward loop. nn.Module intercepts every self.x = value assignment, and only Parameters/submodules/registered buffers get special tracking, everything else falls through to a plain Python attribute. Trained 250 steps on MPS (loss 4.28 to ~2.3), ran "hello world!" through it, and compared the pattern against TransformerLens's. Same underlying mechanism, but noisier and blended instead of GPT-2's clean categories, expected from a tiny, undertrained model. Real previous-token lean showed up across several rows, plus one row where the third "l" in "hello" pulled attention back to an earlier "l" instead of just the immediately previous token, an early hint of the same-token-recall shape real induction heads have, not claiming it's an actual induction head at 250 steps.
Ablated head 6 to test whether its attention pattern is actually load-bearing, not just descriptive. Used a TransformerLens hook on blocks.0.attn.hook_pattern to zero out head 6's slice mid-forward-pass, reran on "a dog and a cat". Top prediction didn't change ('$\n$' both times), and its probability barely moved, 0.144 to 0.151, if anything slightly up with the head removed. result: a head can have a clean, consistent attention pattern and still not meaningfully affect the model's output, since the proj layer and everything downstream decide how much any given head's contribution actually matters. Caveat: only tested one head, one sentence, one query position, doesn't rule out head 6 mattering elsewhere.
Also did LeetCode today, 3 total, meeting the daily minimum. 1768. Merge Strings Alternately: first attempt conflated "the first string" with "the string that needs a bounds check," when what actually matters is which string is shorter, not which one comes first in the alternation. Caught with a concrete counterexample (word1="abcd", word2="pq", where word1 is the longer one) that broke the original logic. Also assumed indexing past a string's length would return None, it raises IndexError instead, confirmed by testing "pq"[2] directly. Fixed the check-before-access ordering, then found a real bug in the else branch (was appending word2 before word1, violating "always start with word1"). Refined into a helper function using a closure to read i from the enclosing loop scope, then finally collapsed the whole if/else into one unified loop that checks both strings' bounds independently every iteration, no branching on which is longer at all. 389. Find the Difference: got the frequency-count concept right immediately (counter from t, decrement using s, the surviving count of 1 marks the answer, correctly handles a duplicate-letter case by tracking counts not just presence), but the code had three real bugs. computed dict1.get(i, 0) + 1 without ever assigning it back to dict1, so the dict silently stayed empty. dict1[j] -= 1 then crashed with a KeyError since the key had never actually been created. returned a raw generator expression instead of pulling a value out of it, fixed with next(generator, default). 169. Majority Element (Claude-picked, Boyer-Moore voting): reasoned through the actual algorithm via guided hints, not told outright. Derived the candidate + count "tug-of-war" framing myself off one metaphor hint (+1 on match, -1 on mismatch), then connected that count hitting zero should trigger picking a new candidate, and that the exact same rule elegantly handles picking the very first candidate too if count starts at 0, no special case needed. First code draft manually pre-set candidate = nums[0] before the loop, which works but reintroduces a special case, refined afterward into the fully unified version.

**Doubts I had today, sorted out:**
- nn.Module's __setattr__ interception. only nn.Parameter, submodules, and explicitly registered buffers get special tracking (state_dict, .to(device), gradients), a plain tensor assigned to self just falls through to ordinary Python attribute behavior
- register_buffer is for persistent model state that doesn't change between forward passes (like the tril mask), not for activations like attention weights that get recomputed every single call. tried to force attn_weights into a buffer first, wrong tool
- .detach() vs torch.no_grad(). no_grad stops a whole block of code from building a graph at all, .detach() takes a tensor already attached to a graph and gives back a copy severed from it, without disturbing the original graph elsewhere
- Attention pattern is not the same as causal importance. a head can look at something consistently every forward pass and still contribute almost nothing to the final prediction if downstream layers don't weight its output heavily, confirmed directly with the head 6 ablation result
- "Weights" specifically means the model's learned parameters, not the same thing as "attention pattern," which is a fresh activation computed every forward pass and isn't stored anywhere in the model
- Why "the first string" and "the string that needs a bounds check" aren't the same thing, which one needs checking depends on which is actually shorter, not on alternation order
- Python string/list indexing past the end raises IndexError, doesn't return None like a dict's .get() would for a missing key
- How closures actually work, a nested function looks up a variable from its enclosing scope at call time, not at definition time, which is why merge(word) could use i without taking it as a parameter and still get the current loop value
- dict1.get(key, 0) + 1 computes a value but doesn't store anything, has to be assigned back with dict1[key] = ... or the dict never actually changes
- A generator expression in parentheses isn't a value on its own, need next(generator, default) to actually pull one item out of it
- Boyer-Moore voting algorithm: candidate/count as a tug-of-war, +1 on match/-1 on mismatch, and count hitting 0 as the single unified trigger for both picking the first candidate and replacing an exhausted one

## August 8
Read the RAG paper (Lewis et al. 2020) abstract through Section 1 (Introduction), stopping right at Figure 1 as assigned. Worked out the three real motivations for RAG from the abstract: provenance/traceability, precise editability of knowledge (vs. it being smeared across weights with no way to reach in and fix one fact), and updating knowledge without retraining. Understood extractive (copy-pasted span) vs. generative (synthesized in the model's own words) tasks, and that REALM/ORQA were prior models limited to extractive-only. Worked through RAG-Sequence (same retrieved passage conditions the whole output) vs. RAG-Token (can shift which retrieved passage dominates per output token, same retrieved candidate set either way, no re-retrieval mid-generation), and confirmed a RAG generator still keeps its own parametric memory, retrieval adds a second source rather than replacing it.
Real deep dive on FAISS/vector DB mechanics after initial confusion, several genuine corrections along the way. worked out that FAISS only stores vectors plus internal IDs, never the original text, that LangChain's docstore is a separate structure mapping ID back to text directly (not through vectors again), that pooling combines a single chunk's internal token-vectors into one vector for that chunk only (never combines separate chunks together), and that FAISS holds vectors in memory only, no automatic persistence unless explicitly saved to disk, same failure mode as today's earlier v2.py training runs that were lost since no checkpoint was ever saved.
Prepped mini-rag/docs/ with the Attention paper, transformer_explainer.md, and a cleaned copy of Anthropic's "Core views on AI safety" post. Real conversation about project scope: confirmed this RAG mini-project is a fluency-building exercise for the tools, not the resume-defining project, that's the Week 7-8 public project per the plan, so no need to inflate this one's scope trying to make it "big enough."
LeetCode tonight: 28. Find the Index of the First Occurrence in a String (brute force, self-picked) and 704. Binary Search (recursive, Claude-picked for pattern coverage). Real bugs in both, both eventually correct.
strStr: off-by-one in the bounds check that would've silently rejected a valid match sitting right at the very end of the haystack. Missing return statements on both the success and not-found paths, function returned None for everything at first. Success-detection check was inside the inner loop instead of after it, meaning single-character needles never got detected since the inner loop never executes for them, fixed by moving the check after the loop and correcting the target value it compares against.
Binary Search: reinforced the exact slicing-loses-the-original-index lesson from Valid Palindrome a few days back, applied fresh here. First recursive attempt sliced the array on every call, losing absolute indices, also had a real off-by-one in the slice upper bound (nums[middle+1:high] silently excludes the actual last valid index, causing a crash when searching for the last element in the array). Rebuilt using low/high as recursive parameters against the same original array instead of slicing. Hit Time Limit Exceeded three separate times from getting the midpoint formula wrong, dropping low from the calculation instead of adding it back after halving the gap: (high-low)//2, then (high-1)//2, then (high-1-low)//2, only low+(high-low)//2 is actually correct. Even after fixing that, hit a second infinite loop from a missing base case for an empty search range (low>=high), adding that finally closed it out.

**Doubts I had today, sorted out:**
- RAG's real motivations beyond provenance: precise knowledge editability, and updating knowledge without retraining
- RAG-Sequence vs. RAG-Token, same retrieved document conditions the whole output vs. can shift which document dominates per output token, same retrieved candidate set either way
- FAISS stores only vectors plus IDs, never the original text, LangChain's docstore is a separate structure mapping ID straight back to text
- Pooling combines a chunk's internal token vectors into one vector for that chunk, never combines separate chunks together
- FAISS is in-memory only unless explicitly saved to disk, same failure mode as an unsaved model checkpoint
- Slicing an array inside recursion loses the original absolute index, since the sliced sub-array restarts its own indexing from 0, passing low/high bounds against the same original array avoids this entirely
- The midpoint of a range is low plus half the gap, not just half the gap alone, since half the gap on its own is a distance, not a position
- Binary search needs an explicit base case for an empty range (low>=high), or it can recurse forever on a target that isn't in the array
- Binary search's implementation subtleties (midpoint math, termination conditions) are famously easy to get wrong even for people who understand the algorithm cold, Java's own standard library shipped a real bug in this exact area for about nine years