# Transformer

A PyTorch implementation of the Transformer architecture from scratch to understand how it works under the hood.

## Overview

This project implements a complete Transformer-based language model using PyTorch. The implementation includes all key components of the Transformer architecture: multi-head self-attention, feed-forward networks, positional encoding, and layer normalization. The model is trained on character-level text data to generate coherent sequences.

## Architecture Components

### Core Modules

- **Head (Single Attention Head)**: Implements scaled dot-product attention with query, key, and value projections. Includes causal masking to prevent the model from attending to future tokens.

- **MultiHeadAttention**: Runs multiple attention heads in parallel and concatenates their outputs for a linear projection, allowing the model to attend to different representation subspaces.

- **FeedForward**: A two-layer feed-forward network with ReLU activation and dropout, applied independently to each position.

- **Block**: A Transformer block combining multi-head self-attention, feed-forward network, layer normalization, and residual connections.

- **BigramLanguageModel**: The full Transformer model with token embeddings, positional embeddings, stacked Transformer blocks, and a language modeling head for next-token prediction.

## Key Features

- **Multi-head Self-Attention**: 6 attention heads allowing the model to focus on different parts of the input
- **Causal Masking**: Prevents information flow from future tokens, making it suitable for language modeling
- **Positional Encoding**: Learnable positional embeddings to capture token positions in the sequence
- **Residual Connections**: Enables deeper networks to train effectively
- **Dropout**: Applied throughout for regularization
- **Character-level Training**: Learns to predict the next character given a context window

## Hyperparameters

- **Embedding Dimension**: 384
- **Number of Attention Heads**: 6
- **Number of Transformer Blocks**: 6
- **Block Size (Context Window)**: 256
- **Batch Size**: 64
- **Learning Rate**: 3e-4
- **Dropout**: 0.2

## Training

The model is trained using cross-entropy loss with the AdamW optimizer. It learns to predict the next character in a sequence given the previous characters, similar to how large language models work.

## Usage

Run the training script:

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

This implementation helped understand:
- How attention mechanisms work in practice
- The role of positional encoding in Transformers
- How multiple attention heads capture different patterns
- The importance of residual connections and layer normalization
- End-to-end training of a generative language model
