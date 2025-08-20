# Simple ViT
This repository presents a [*Vision Tranformer (ViT)*](https://en.wikipedia.org/wiki/Vision_transformer) implementation.

For more information about Transformer Model I recommend [Simple Transformer](https://github.com/Bengal1/Simple-Transformer).

## Requirements
- [![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/) <br/>
- [![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/) <br/>

## Vision Transformer
<img align="right" width="300" alt="ViT Architecture" src="https://github.com/user-attachments/assets/fafbde82-d337-45ac-bdbc-31c6b6df3b62" />

The Vision Transformer (ViT) is a deep learning architecture that adapts the Transformer, originally developed for natural language processing, to image recognition tasks. Introduced by Dosovitskiy et al. in “An Image is Worth 16x16 Words” (2020), ViT replaces traditional convolutional feature extractors with a sequence of image patches processed by self-attention. This approach demonstrated that, with sufficient data and compute, Transformers can outperform convolutional neural networks (CNNs) in computer vision benchmarks, paving the way for a broad family of vision transformer models.<br/>

In practice, ViT transforms an image into a sequence of smaller patches, which are then processed using the same self-attention mechanism that made Transformers successful in language tasks. By modeling relationships between patches directly, ViT captures both local details and long-range dependencies within an image, offering a flexible alternative to the strictly hierarchical representations of CNNs. Positional information is incorporated to maintain awareness of spatial structure, and a dedicated representation is used for classification. This design shifts the focus from handcrafted inductive biases toward a more data-driven approach, where the model learns to interpret visual structure primarily from large-scale training data.

### Patch Embedding
<img align="right" width="400" alt="patch_embedding_data" src="https://github.com/user-attachments/assets/f3d1a1cd-b3c8-4604-a23b-087f2caaadd4" />

A key step in the Vision Transformer (ViT) is the patch embedding stage, which transforms an image into a sequence suitable for a Transformer. Instead of processing pixels individually or relying on convolutional filters, the input image is divided into fixed-size patches (for example, 16×16 pixels). Each patch is then flattened into a vector and projected through a linear layer to a chosen embedding space. The result is a sequence of patch embeddings that can be treated similarly to word tokens in natural language processing, allowing the Transformer to apply self-attention mechanisms across the entire image.<br/>

### CLS Token
The [CLS] token is a learnable embedding prepended to the sequence of patch embeddings in a Vision Transformer. Its primary purpose is to serve as a global representation of the entire image. During the forward pass, the Transformer encoder processes the sequence of patch embeddings along with the [CLS] token, allowing the self-attention mechanism to integrate information from all patches into this special token. After the final encoder block, the [CLS] token contains a summary of the image’s content and is typically fed into the classification head to produce the output logits. By using the [CLS] token in this way, ViT can perform classification based on a single learned representation rather than aggregating information from all patch embeddings.

### Positional Encoding
Since Transformers process input sequences without any inherent notion of order, it is necessary to provide information about the position of each patch in the image. In the Vision Transformer, this is achieved through positional encoding, which adds a vector to each patch embedding to indicate its location within the image. Unlike the fixed sinusoidal encodings used in the original Transformer for NLP, ViT often uses learnable positional embeddings, which are initialized randomly and updated during training. These learnable embeddings allow the model to adaptively encode spatial relationships between patches, helping the self-attention mechanism capture both local and global structure in the image.

### Transformer Encoder
<img align="right" width="340" alt="Encoder" src="https://github.com/user-attachments/assets/a0b78aca-5f38-4a85-8708-9c3c3bb0e85e" />

The Transformer encoder is a fundamental component of the Vision Transformer (ViT), responsible for processing the sequence of patch embeddings and capturing relationships between them. Each encoder block contains a multi-head self-attention layer, which allows the model to weigh the importance of each patch relative to all others, followed by a feed-forward network (MLP) that transforms the representations. Residual connections and layer normalization are applied throughout to stabilize training and improve gradient flow. By stacking multiple encoder blocks, the Transformer encoder can build complex, high-level representations of the image, integrating both local and global information for downstream tasks such as classification.

#### Attention
Attention is a core mechanism in transformers that allows the model to selectively focus on the most relevant parts of an input sequence when making predictions. Instead of processing information uniformly, attention assigns weights to different elements, enabling the network to capture both local and long-range dependencies. In the context of Vision Transformers (ViTs), self-attention is applied directly to image patches, treating them as a sequence of tokens similar to words in natural language processing. This mechanism allows each patch to attend to every other patch, capturing global spatial relationships across the image. Unlike convolutional operations, which have a fixed receptive field, self-attention provides a flexible and adaptive way of modeling dependencies, making it particularly powerful for understanding complex visual structures. In Vision Transformer we apply Multi-Head Self Attention
Given an input sequence of tokens (patch embeddings) $`X∈ℝ^{N×D}`$ where $`N`$ is the number of patches and $`D`$ is the embedding dimension, self-attention computes interactions between all tokens as follows:
1. **Linear projections for queries, keys, and values:**
   
$$
  X·W_{Q} = Q &ensp; ; &ensp; X·W_{K} = K &ensp; ; &ensp; X·W_{V} = V
$$

where $`W_{Q}, W_{K}, W_{V} ∈ℝ^{D×d}`$ are learnable weight matrices, and $`d`$ is the attention head dimension.<br/>

2. **Scaled dot-product attention:**

```math
Attention(Q,K,V) = Softmax \Bigg(\frac{Q K^{T}}{\sqrt{d}} \Bigg)·V
```
* $`Q K^{T}∈ℝ^{N×N}`$ computes similarity between every pair of tokens.
* $`\sqrt{d}`$ is a scaling factor to stabilize gradients.
* The softmax converts similarities into attention weights.
  
3. **Multi-head attention (concatenation of the heads):**
```math
MultiHead-Attention = Concat(head_1,...,head_h)·W_{O}
```
* Multiple attention heads allow the model to capture different types of interactions.
* $`W_{O} ∈ℝ^{hd×D}`$ projects concatenated outputs back to the embedding dimension.

For more details information about *Attention Mechanism* see [Simple Transformer](https://github.com/Bengal1/Simple-Transformer).

#### Feed Forward
<img align="right" width="400"  src="https://github.com/user-attachments/assets/484983aa-a374-4d71-bca1-f94467502650">

The feed-forward network (FFN) in the Vision Transformer (ViT) is a crucial component of each encoder block. It consists of two fully connected layers with a non-linear activation function, often GELU (Gaussian Error Linear Unit), applied between them. Unlike self-attention, which enables tokens to exchange information globally, the FFN operates on each token independently, refining and transforming its representation in a higher-dimensional space. This allows the model to capture more complex, non-linear relationships within the data. In ViT, the FFN complements self-attention by enhancing the expressive power of the patch embeddings, ensuring that both global context and token-wise transformations contribute to the learned image representation.

```math
y = f(W_{1}·x+b_{1})·W_{2} + b_{2}
```
Where:
* ***$`x`$*** is the input vector.
* ***$`W_i`$*** is the weight matrix of layer *i*.
* ***$`b_i`$*** is the bias vector of layer *i*.
* ***$`f`$*** is the activation function - GELU.

#### Layer Normalization
<img align="right" width="250"  src="https://github.com/user-attachments/assets/a1434118-a1d7-4a40-a35e-14b922ee0db4">

*Layer Normalization* is used to stabilize and accelerate training by normalizing the inputs to each layer.<br/>
For each input vector (for each token in a sequence), subtract the mean and divide by the standard deviation of the vector's values. This centers the data around 0 with unit variance:
```math
\hat{x} = \frac{(x - μ)}{\sqrt{σ^{2} + ε}}
```
where *μ* is the mean and *σ* is the standard deviation of the input vector.<br/><br/>
Then apply scaling (gamma) and shifting (beta) parameters (trainable):

* *γ* (scale): A parameter to scale the normalized output.<br/>
* *β* (shift): A parameter to shift the normalized output.<br/>

```math
⇨  y = γ·\hat{x} + β
```

## Training and Optimization
### Adam Optimizer
The Adam optimization algorithm is an extension to stochastic gradient descent (SGD). Unlike SGD, The method computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients Adam combines the benefits of two other methods: momentum and RMSProp.

#### Adam Algorithm:
* $`\theta_t`$​ : parameters at time step *t*.
* $`\beta_1,\beta_2​`$: exponential decay rates for moments estimation.
* $`\alpha`$ : learning rate.
* $`\epsilon`$ : small constant to prevent division by zero.
* $`\lambda`$ : weight decay coefficient. <br/>

1. Compute gradients:

$$
g_t = \nabla_{\theta} J(\theta_t)
$$

2. Update first moment estimate (mean):

$$
m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t
$$

3. Update second moment estimate (uncentered variance):

$$
v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2
$$

4. Bias correction:

$$
\hat{v}_t = \frac{v_t}{1 - \beta_2^t} \quad ; \quad \hat{m}_t = \frac{m_t}{1 - \beta_1^t}
$$

5. Update parameters:

$$
\theta_{t+1} = \theta_t - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

* In our model *Weight decay* is applied (decoupled):

$$
\theta_{t+1} ← \theta_{t+1} - \alpha \cdot \lambda \cdot \theta_t 
$$


### Cross-Entropy Loss Function
This criterion computes the cross entropy loss between input logits and target. Loss function is a function that maps an event or values of one or more variables onto a real number intuitively representing some "loss" associated with the event. The Cross Enthropy Loss function is commonly used in classification tasks both in traditional ML and deep learning. It compares the predicted probability distribution over classes (logits) with the true class labels and penalizes incorrect or uncertain predictions.

$$
Loss = - \sum_{i=1}^{C} y_i \log(\hat{y}_i)
$$

Where:
* $`C`$  is the number of classes.
* $`y_i`$​  is the true probability for class *i* (usually 1 for the correct class and 0 for others).
* $`\hat{y}_i`$  is the predicted probability for class *i*.


## ViT vs CNN
Convolutional Neural Networks (CNNs), first demonstrated in LeNet-5 (LeCun et al., 1998) and popularized by AlexNet (2012), dominated computer vision for decades. They rely on convolutional filters applied to local receptive fields, pooling for downsampling, and fully connected layers for classification. This design encodes strong inductive biases: locality (features are learned from neighboring pixels) and translation equivariance (patterns can be recognized regardless of position). Variants like VGG, ResNet, and DenseNet advanced CNNs by increasing depth and introducing innovations such as residual connections.

The Vision Transformer (ViT), introduced by Dosovitskiy et al. (2020), replaces convolutions with a pure Transformer encoder. An image is split into fixed-size patches (e.g., 16×16), flattened, linearly projected into embeddings, and combined with positional encodings. These are processed by Multi-Head Self-Attention (MHSA), which models global dependencies between all patches in parallel, something CNNs only capture gradually via deeper layers. A special [CLS] token aggregates global features for classification.

In architecture, CNNs build hierarchical representations through stacked convolutions, while ViTs operate directly in patch-embedding space, with flexible receptive fields determined by attention. Practically, CNNs train well on moderate datasets due to their inductive biases, whereas ViTs usually require large-scale pretraining and strong regularization but can outperform CNNs when sufficient data is available.

## Data


## Evaluation


## Reference
[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805?utm_source=chatgpt.com)

[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929?utm_source=chatgpt.com)

[Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
