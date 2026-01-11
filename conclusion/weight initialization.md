# Weight Initialization in Deep Learning

Weight initialization determines how neural network weights are set **before training begins**.  
Good initialization prevents **vanishing gradients**, **exploding gradients**, and **slow convergence**.

---

## Why Initialization Matters

If weights are:
- **Too small** → gradients vanish → network stops learning
- **Too large** → gradients explode → unstable training
- **All zeros** → symmetry problem → neurons learn the same thing

🎯 **Goal:**  
Maintain stable variance of activations and gradients across layers.

---

## Key Terms

- **fan_in**  → number of input connections
- **fan_out** → number of output connections

---

## Xavier (Glorot) Initialization

Designed for **linear, tanh, sigmoid** activations and **attention layers**.

### Xavier Uniform
```python
nn.init.xavier_uniform_(W)

W∼U(−6fanin+fanout,  6fanin+fanout)
W∼U(−fanin​+fanout​6​
​,fanin​+fanout​6​
​)
Xavier Normal

nn.init.xavier_normal_(W)

W∼N(0,  2fanin+fanout)
W∼N(0,fanin​+fanout​2​
​)
When to Use

    Transformers

    Attention (Query, Key, Value projections)

    Linear layers

    Mamba / SSM models

He (Kaiming) Initialization

Designed for ReLU-based networks.
He Normal (most common)

nn.init.kaiming_normal_(W, nonlinearity="relu")

W∼N(0,  2fanin)
W∼N(0,fanin​2​
​)
He Uniform

nn.init.kaiming_uniform_(W, nonlinearity="relu")

When to Use

    CNNs

    Deep MLPs

    ReLU / LeakyReLU

❌ Not recommended for attention layers
LeCun Initialization

Used with SELU activations for self-normalizing networks.

nn.init.normal_(W, mean=0, std=1/\sqrt{fan_{in}})

Orthogonal Initialization

nn.init.orthogonal_(W)

    Preserves vector norms

    Stable gradients

    Common in RNNs and SSMs

Truncated Normal Initialization

Used in many Transformer models (BERT, ViT).

nn.init.trunc_normal_(W, std=0.02)

Prevents rare large weights.
Bias Initialization

nn.init.zeros_(bias)

✔ Biases are usually initialized to zero
❌ Weights should never be all zeros
Comparison Table
Initialization	Best Use Case	Activation
Xavier	Attention, Linear	tanh, sigmoid
He	CNNs, Deep Nets	ReLU
LeCun	Self-normalizing nets	SELU
Orthogonal	RNNs, SSMs	Any
Truncated Normal	Transformers	GELU
PyTorch Defaults
Layer	Default Initialization
nn.Linear	Kaiming Uniform
nn.Conv2d	Kaiming Uniform
Attention Layers	Custom (recommended)
Recommended for ST-Mamba / Attention

nn.init.xavier_uniform_(self.W_query.weight)
nn.init.xavier_uniform_(self.W_key.weight)
nn.init.xavier_uniform_(self.W_value.weight)

✔ Stable attention scores
✔ Prevents softmax saturation
✔ Faster convergence
One-Line Summary

    Weight initialization decides whether your network starts learning smoothly, chaotically, or not at all.


---

If you want, next I can:
- Add **mathematical derivations**
- Add **diagrams**
- Customize it for **ST-Mamba specifically**
- Convert this into **notes for exams**

Just say the word 👍


ChatGPT can make mistakes. Check important info. See Cookie Preferences.
