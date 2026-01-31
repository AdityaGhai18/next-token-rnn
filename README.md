# RNN Next-Token Prediction

A vanilla RNN language model built from scratch in PyTorch. Trained on sci-fi movie screenplays to generate new screenplay text character-by-character.

## Why I Built This

Wanted to deeply understand how RNNs actually work under the hood—not just calling `nn.RNN` but implementing the recurrence manually. This meant writing out the hidden state update equations myself, dealing with vanishing gradients firsthand, and seeing how architectural choices (LayerNorm, gradient clipping, initialization) affect training stability.

The goal was never SOTA results, just learning. And it was pretty cool watching the model go from outputting random garbage to generating coherent screenplay-style text with proper formatting (INT., EXT., character names, dialogue) from a very basic character tokenizer. I wanted to understand the true need for the transformer architecture so a big motivation of this was to see how far i can push the context length for this.

## Project Structure

```
├── train.py        # Model definition + training loop
├── dataset.py      # Dataset class with sliding window
├── tokenizer.py    # Character-level tokenizer
├── inference.py    # CLI for generating text
├── data/           # Movie screenplay .txt files
└── checkpoints/    # Saved model weights
```
