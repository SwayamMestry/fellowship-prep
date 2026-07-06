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