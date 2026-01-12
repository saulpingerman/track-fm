# Experiment 7: Decoder Transformer Backbone

## Question

Does a decoder transformer (causal/autoregressive attention) perform better or worse than an encoder transformer (bidirectional attention) for trajectory prediction?

## Background

Experiment 1 used an encoder transformer where each position can attend to all other positions (bidirectional attention). This experiment tests whether causal attention (each position only sees previous positions) is more appropriate for sequential trajectory data.

### Key Differences

| Aspect | Encoder (Exp 1) | Decoder (Exp 7) |
|--------|-----------------|-----------------|
| Attention | Bidirectional | Causal (autoregressive) |
| Position i sees | All positions | Positions 0 to i only |
| Similar to | BERT | GPT |

## Hypothesis

Causal attention may be more natural for trajectory prediction because:
1. Trajectories are inherently sequential (time flows forward)
2. The model should learn to predict "what comes next" based on "what came before"
3. This mirrors how real-world tracking systems operate

Alternatively, bidirectional attention may be better because:
1. Looking at the full trajectory context might help extract velocity more accurately
2. The encoder can see the entire motion pattern at once

## Method

1. Implement a decoder transformer using causal masking (upper triangular attention mask)
2. Run all loss functions from Experiment 1 on both encoder and decoder
3. Compare prediction errors

## Implementation Details

The decoder uses the same architecture as the encoder but with a causal attention mask:

```python
def generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Position i can only attend to positions <= i."""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.bool()
```

This is applied to `nn.TransformerEncoder` (which is how GPT-style "decoder-only" models are implemented in PyTorch).

## How to Run

```bash
cd experiments/07_decoder_transformer
python3 run_experiment.py
```

## Expected Output

The experiment will produce:
- Side-by-side comparison of encoder vs decoder for all loss functions
- Visualization plots for both architectures
- Summary table showing which architecture performs better

## Results

(To be filled in after running the experiment)

| Method | Encoder Error | Decoder Error | Winner |
|--------|---------------|---------------|--------|
| Regression | | | |
| NLL | | | |
| Soft Target | | | |
| Curriculum | | | |
| Mixed Loss | | | |
