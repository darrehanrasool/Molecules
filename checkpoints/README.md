# Model Checkpoints Repository

This directory archives serialized PyTorch model weights and state dicts:

- `best/`: Checkpoints achieving the highest validation MRR@25 score.
- `latest/`: Most recent epoch state for seamless training resumption.
- `periodic/`: Checkpoints saved at fixed epoch intervals.
- `metadata/`: JSON training logs containing epoch loss, learning rates, and validation metrics.
