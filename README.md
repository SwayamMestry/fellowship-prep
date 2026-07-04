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