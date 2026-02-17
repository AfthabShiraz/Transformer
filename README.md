# Transformer

I built this project to create a PyTorch implementation of the Transformer architecture from scratch so I could understand how it works under the hood.

## Overview

I implemented a complete Transformer-based language model using PyTorch. My implementation includes all the key components of the Transformer architecture: multi-head self-attention, feed-forward networks, layer normalization, and residual connections.

## Architecture Components

### Core Modules

- **Head (Single Attention Head)**: I implemented scaled dot-product attention with query, key, and value projections. I included causal masking to prevent the model from attending to future tokens.

- **MultiHeadAttention**: I developed this to run multiple attention heads in parallel and concatenate their outputs for a linear projection, allowing the model to attend to different representation subspaces.

- **FeedForward**: I built a two-layer feed-forward network with ReLU activation and dropout, applied independently to each position.

- **Block**: I created a Transformer block that combines multi-head self-attention, feed-forward network, layer normalization, and residual connections.

- **BigramLanguageModel**: I built the full Transformer model with token embeddings, positional embeddings, stacked Transformer blocks, and a language modeling head for next-token prediction.

## Key Features

- **Multi-head Self-Attention**: I configured 6 attention heads allowing the model to focus on different parts of the input
- **Causal Masking**: I implemented this to prevent information flow from future tokens, making it suitable for language modeling
- **Positional Encoding**: I used learnable positional embeddings to capture token positions in the sequence
- **Residual Connections**: I added these to enable deeper networks to train effectively
- **Dropout**: I applied this throughout for regularization
- **Character-level Training**: My model learns to predict the next character given a context window

## Hyperparameters

I tuned the following hyperparameters for my implementation:

- **Embedding Dimension**: 384
- **Number of Attention Heads**: 6
- **Number of Transformer Blocks**: 6
- **Block Size (Context Window)**: 256
- **Batch Size**: 64
- **Learning Rate**: 3e-4
- **Dropout**: 0.2

## Training

I trained the model using cross-entropy loss with the AdamW optimizer. It learns to predict the next character in a sequence given the previous characters, similar to how large language models work.

## Usage

To run my training script:

```bash
python train.py
```

The script will:
1. Load text data from `input.txt`
2. Build character-level vocabularies
3. Create training and validation splits
4. Train the Transformer model
5. Generate new text sequences using the trained model

## Learning Outcomes

Through this implementation, I gained a deeper understanding of:
- How attention mechanisms work in practice
- The role of positional encoding in Transformers
- How multiple attention heads capture different patterns
- The importance of residual connections and layer normalization
- End-to-end training of a generative language model