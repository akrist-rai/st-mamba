
---

````md
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
````

[
W \sim U\left(-\sqrt{\frac{6}{fan_{in}+fan_{out}}},;
\sqrt{\frac{6}{fan_{in}+fan_{out}}}\right)
]

### Xavier Normal

```python
nn.init.xavier_normal_(W)
```

[
W \sim \mathcal{N}\left(0,;\sqrt{\frac{2}{fan_{in}+fan_{out}}}\right)
]

### When to Use

* Transformers
* Attention (Query, Key, Value projections)
* Linear layers
* Mamba / SSM models

---

## He (Kaiming) Initialization

Designed for **ReLU-based networks**.

### He Normal (most common)

```python
nn.init.kaiming_normal_(W, nonlinearity="relu")
```

[
W \sim \mathcal{N}\left(0,;\sqrt{\frac{2}{fan_{in}}}\right)
]

### He Uniform

```python
nn.init.kaiming_uniform_(W, nonlinearity="relu")
```

### When to Use

* CNNs
* Deep MLPs
* ReLU / LeakyReLU

❌ Not recommended for attention layers

---

## LeCun Initialization

Used with **SELU** activations for self-normalizing networks.

```python
nn.init.normal_(W, mean=0, std=1/\sqrt{fan_{in}})
```

---

## Orthogonal Initialization

```python
nn.init.orthogonal_(W)
```

* Preserves vector norms
* Stable gradients
* Common in RNNs and SSMs

---

## Truncated Normal Initialization

Used in many Transformer models (BERT, ViT).

```python
nn.init.trunc_normal_(W, std=0.02)
```

Prevents rare large weights.

---

## Bias Initialization

```python
nn.init.zeros_(bias)
```

✔ Biases are usually initialized to zero
❌ Weights should never be all zeros

---

## Comparison Table

| Initialization   | Best Use Case         | Activation    |
| ---------------- | --------------------- | ------------- |
| Xavier           | Attention, Linear     | tanh, sigmoid |
| He               | CNNs, Deep Nets       | ReLU          |
| LeCun            | Self-normalizing nets | SELU          |
| Orthogonal       | RNNs, SSMs            | Any           |
| Truncated Normal | Transformers          | GELU          |

---

## PyTorch Defaults

| Layer            | Default Initialization |
| ---------------- | ---------------------- |
| `nn.Linear`      | Kaiming Uniform        |
| `nn.Conv2d`      | Kaiming Uniform        |
| Attention Layers | Custom (recommended)   |

---

## Recommended for ST-Mamba / Attention

```python
nn.init.xavier_uniform_(self.W_query.weight)
nn.init.xavier_uniform_(self.W_key.weight)
nn.init.xavier_uniform_(self.W_value.weight)
```

✔ Stable attention scores
✔ Prevents softmax saturation
✔ Faster convergence

---

## One-Line Summary

> **Weight initialization decides whether your network starts learning smoothly, chaotically, or not at all.**



```

---

```
