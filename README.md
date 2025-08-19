# Simple ViT
This repository presents a [*Vision Tranformer (ViT)*](https://en.wikipedia.org/wiki/Vision_transformer) implementation.

## Vision Transformer
The Vision Transformer (ViT) is a deep learning architecture that adapts the Transformer, originally developed for natural language processing, to image recognition tasks. Introduced by Dosovitskiy et al. in “An Image is Worth 16x16 Words” (2020), ViT replaces traditional convolutional feature extractors with a sequence of image patches processed by self-attention. This approach demonstrated that, with sufficient data and compute, Transformers can outperform convolutional neural networks (CNNs) in computer vision benchmarks, paving the way for a broad family of vision transformer models.

### Patch Embedding
A key step in the Vision Transformer (ViT) is the patch embedding stage, which transforms an image into a sequence suitable for a Transformer. Instead of processing pixels individually or relying on convolutional filters, the input image is divided into fixed-size patches (for example, 16×16 pixels). Each patch is then flattened into a vector and projected through a linear layer to a chosen embedding space. The result is a sequence of patch embeddings that can be treated similarly to word tokens in natural language processing, allowing the Transformer to apply self-attention mechanisms across the entire image.

### Positional Encoding
Since Transformers process input sequences without any inherent notion of order, it is necessary to provide information about the position of each patch in the image. In the Vision Transformer, this is achieved through positional encoding, which adds a vector to each patch embedding to indicate its location within the image. Unlike the fixed sinusoidal encodings used in the original Transformer for NLP, ViT often uses learnable positional embeddings, which are initialized randomly and updated during training. These learnable embeddings allow the model to adaptively encode spatial relationships between patches, helping the self-attention mechanism capture both local and global structure in the image.

### Transformer Encoder
The Transformer encoder is a fundamental component of the Vision Transformer (ViT), responsible for processing the sequence of patch embeddings and capturing relationships between them. Each encoder block contains a multi-head self-attention layer, which allows the model to weigh the importance of each patch relative to all others, followed by a feed-forward network (MLP) that transforms the representations. Residual connections and layer normalization are applied throughout to stabilize training and improve gradient flow. By stacking multiple encoder blocks, the Transformer encoder can build complex, high-level representations of the image, integrating both local and global information for downstream tasks such as classification.

#### Attention


#### Feed Forward


#### Layer Normalization


### ViT vs CNN


## Data


## Evaluation


## Reference


