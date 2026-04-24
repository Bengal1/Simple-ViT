# ViT vs CNN

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

A comparative study of Vision Transformers (ViT) and Convolutional Neural Networks (CNNs) for image classification. This project explores how model performance evolves across datasets of increasing complexity, highlighting the transition from CNN dominance on simple tasks to ViT advantages on more challenging visual data.

This repository also showcases the [*Vision Transformer (ViT)*](https://en.wikipedia.org/wiki/Vision_transformer), providing insight into its core architectural ideas, training dynamics, and its ability to capture global relationships through self-attention. It offers a practical view of the strengths and limitations of ViTs, and how they differ from traditional convolutional approaches.


## 🚀 Quick Start

Run a training experiment in one command:
```bash
python main.py --model vit --dataset tiny_imagenet
```

## 📊 Results

| Model | Metrics           | MNIST      | CIFAR-10   | Tiny ImageNet |
|-------|-------------------|------------|------------|---------------|
| ViT   | Parameters        | 18,955,274 | 18,979,338 |  19,175,112   |
|       | Test Accuracy (%) |   93.27    |   62.26    |     27.84     |
|       | Test Loss         |   0.6838   |  1.40396   |    3.6533     |
| CNN   | Parameters        |  843,850   | 1,204,874  |   6,545,224   |
|       | Test Accuracy (%) |   99.35    |    72.28   |     22.67     |
|       | Test Loss         |   0.5305   |   1.1652   |    3.9591     |


## Training & Execution
Follow the steps below to set up and run the project.

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run training
```bash
python main.py --model <model> --dataset <dataset>
```

### Configuration
Choose a model and dataset:

**Models:**
- `vit` — Vision Transformer
- `cnn` — Convolutional Neural Network

**Datasets:**
- `mnist` — MNIST dataset
- `cifar10` — CIFAR-10 dataset
- `tiny_imagenet` — Tiny ImageNet dataset

### Output
After training:
- Metrics are saved to `results/`
- Training and validation plots are generated automatically
- GPU is used if available

### Notes
- Datasets are downloaded automatically if not present
- Hyperparameters can be modified in `config.py`


## 🧠 Vision Transformer
<img align="right" width="300" alt="ViT Architecture" src="https://github.com/user-attachments/assets/41934e26-ecd0-4aec-89a3-b78fee241ebb" />

The Vision Transformer (ViT) is a deep learning architecture that adapts the Transformer, originally developed for natural language processing, to image recognition tasks. Introduced by Dosovitskiy et al. in “An Image is Worth 16x16 Words” (2020), ViT replaces traditional convolutional feature extractors with a sequence of image patches processed by self-attention. This approach demonstrated that, with sufficient data and compute, Transformers can outperform convolutional neural networks (CNNs) in computer vision benchmarks, paving the way for a broad family of vision transformer models.<br/>

In practice, ViT transforms an image into a sequence of smaller patches, which are then processed using the same self-attention mechanism that made Transformers successful in language tasks. By modeling relationships between patches directly, ViT captures both local details and long-range dependencies within an image, offering a flexible alternative to the strictly hierarchical representations of CNNs. Positional information is incorporated to maintain awareness of spatial structure, and a dedicated representation is used for classification. This design shifts the focus from handcrafted inductive biases toward a more data-driven approach, where the model learns to interpret visual structure primarily from large-scale training data.

For more information about Transformer Model I recommend [Simple Transformer](https://github.com/Bengal1/Simple-Transformer).

### Patch Embedding
<img align="right" width="400" alt="patch_embedding_data" src="https://github.com/user-attachments/assets/49aa0282-b19e-4fab-a494-de5e708b8478" />

A key step in the Vision Transformer (ViT) is the patch embedding stage, which transforms an image into a sequence suitable for a Transformer. Instead of processing pixels individually or relying on convolutional filters, the input image is divided into fixed-size patches (for example, 16×16 pixels). Each patch is then flattened into a vector and projected through a linear layer to a chosen embedding space. The result is a sequence of patch embeddings that can be treated similarly to word tokens in natural language processing, allowing the Transformer to apply self-attention mechanisms across the entire image.<br/>

### CLS Token
The [CLS] token is a learnable embedding prepended to the sequence of patch embeddings in a Vision Transformer. Its primary purpose is to serve as a global representation of the entire image. During the forward pass, the Transformer encoder processes the sequence of patch embeddings along with the [CLS] token, allowing the self-attention mechanism to integrate information from all patches into this special token. After the final encoder block, the [CLS] token contains a summary of the image’s content and is typically fed into the classification head to produce the output logits. By using the [CLS] token in this way, ViT can perform classification based on a single learned representation rather than aggregating information from all patch embeddings.

### Positional Encoding
Since Transformers process input sequences without any inherent notion of order, it is necessary to provide information about the position of each patch in the image. In the Vision Transformer, this is achieved through positional encoding, which adds a vector to each patch embedding to indicate its location within the image. Unlike the fixed sinusoidal encodings used in the original Transformer for NLP, ViT often uses learnable positional embeddings, which are initialized randomly and updated during training. These learnable embeddings allow the model to adaptively encode spatial relationships between patches, helping the self-attention mechanism capture both local and global structure in the image.

### Transformer Encoder
<img align="right" width="300" alt="Encoder" src="https://github.com/user-attachments/assets/ce78de70-696e-4968-bf7d-345d23c2bbc1" />

The Vision Transformer (ViT) is built on the Transformer encoder architecture, which processes images as a sequence of patch embeddings. Each encoder block combines multi-head self-attention and a feed-forward network, with normalization and residual connections to stabilize training.

The key component is self-attention, which allows each image patch to interact with every other patch. Unlike convolutional layers that focus on local neighborhoods, self-attention captures global relationships across the entire image. This enables the model to learn long-range dependencies and complex visual patterns more effectively.

By stacking multiple encoder blocks, ViT builds increasingly rich representations of the input, integrating both local details and global context for tasks such as image classification.

For a deeper explanation of the Transformer architecture, see:
[Simple Transformer](https://github.com/Bengal1/Simple-Transformer)


## ViT vs CNN
Convolutional Neural Networks (CNNs), first demonstrated in LeNet-5 (LeCun et al., 1998) and popularized by AlexNet (2012), dominated computer vision for decades. They rely on convolutional filters applied to local receptive fields, pooling for downsampling, and fully connected layers for classification. This design encodes strong inductive biases: locality (features are learned from neighboring pixels) and translation equivariance (patterns can be recognized regardless of position). Variants like VGG, ResNet, and DenseNet advanced CNNs by increasing depth and introducing innovations such as residual connections.<br/>
To learn more about Convolutional Neural Networks (CNNs), I recommend [Simple CNN Guide](https://github.com/Bengal1/Simple-CNN-Guide).

The Vision Transformer (ViT), introduced by Dosovitskiy et al. (2020), replaces convolutions with a pure Transformer encoder. An image is split into fixed-size patches (e.g., 16×16), flattened, linearly projected into embeddings, and combined with positional encodings. These are processed by Multi-Head Self-Attention (MHSA), which models global dependencies between all patches in parallel, something CNNs only capture gradually via deeper layers. A special [CLS] token aggregates global features for classification.

<img align="right" width="400" alt="CNN vs ViT - Receptive Field" src="https://github.com/user-attachments/assets/a81742f9-b714-4a54-b3d9-dc0130135be3" />
In architecture, CNNs build hierarchical representations through stacked convolutions, gradually expanding their receptive fields and excelling at capturing local patterns such as edges and textures. ViTs, on the other hand, operate directly in patch-embedding space, where self-attention provides a global receptive field from the very first layer. This fundamental difference means CNNs have built-in biases for vision, making them data-efficient and effective on smaller datasets, while ViTs rely more heavily on large-scale data to learn spatial relationships that CNNs encode naturally. Consequently, CNNs tend to be more efficient and perform strongly when training data is limited, benefiting from their inductive biases, whereas ViTs scale more effectively with increased model size and dataset availability, often surpassing CNNs in accuracy and adaptability. Additionally, CNNs naturally form structured feature hierarchies that emphasize local detail and are relatively easy to interpret, making them well-suited for tasks like object detection or medical imaging. ViTs, by modeling long-range dependencies from the start, offer greater flexibility in capturing global context, which can lead to stronger performance in complex recognition challenges and transfer learning scenarios.


## Data
The experiments are conducted on three standard image classification datasets of increasing complexity: MNIST, CIFAR-10, and Tiny ImageNet. MNIST consists of simple grayscale handwritten digits, CIFAR-10 introduces low-resolution RGB images of natural objects, and Tiny ImageNet provides higher-resolution images with a large number of classes and greater visual diversity. Together, these datasets enable a progressive evaluation of model performance from simple to more complex visual recognition tasks.<br/>

### MNIST
<img align="right" width="130" alt="mnist" src="https://github.com/user-attachments/assets/5616cd75-337d-4ff1-aa47-98e4af773208" />
MNIST is a grayscale image dataset of handwritten digits (0–9), where each image has a resolution of 28×28 pixels. It contains 60,000 training samples and 10,000 test samples, with relatively low variability and simple patterns. Due to its simplicity and strong local features, it is well-suited for benchmarking basic models and is typically dominated by convolutional neural networks.<br/>

### CIFAR-10
<img align="right" width="130" alt="cifar10" src="https://github.com/user-attachments/assets/9693d488-70f7-4222-a2ea-c11783294d3a" />
CIFAR-10 is a dataset of 32×32 RGB images across 10 object classes such as animals and vehicles. It consists of 50,000 training samples and 10,000 test samples, with moderate variability and more complex visual structure than MNIST. While still relatively small in resolution, it provides a more realistic benchmark for image classification and highlights the limitations of models that rely only on local features.<br/>


### Tiny ImageNet
<img align="right" width="130" alt="tiny_imagenet" src="https://github.com/user-attachments/assets/26f7eb66-2596-4aaf-8434-f1c5c1bc4788" />
Tiny ImageNet is a scaled-down version of ImageNet, containing 64×64 RGB images across 200 classes, with 100,000 training samples and 10,000 validation samples. It introduces higher resolution, greater class diversity, and increased semantic complexity compared to CIFAR-10. This makes it a more challenging dataset where models that capture global context, such as Vision Transformers, begin to show advantages.<br/><br/>

The three datasets differ primarily in image complexity, resolution, and semantic diversity. MNIST contains simple, centered grayscale digits with minimal variation, making it an easy task dominated by local features. CIFAR-10 introduces color, background noise, and object variability at a small resolution, increasing the difficulty and requiring better feature extraction. Tiny ImageNet further increases resolution and the number of classes, adding significant intra-class variation and global structure, which makes the task more challenging and better suited for models that capture broader contextual relationships.<br/>

## Reference
[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805?utm_source=chatgpt.com)

[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929?utm_source=chatgpt.com)

[Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
