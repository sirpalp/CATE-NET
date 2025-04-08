
# CATE-NET: Chaotic Attractor Transition Ensemble Network

This repository contains code for modeling EEG brain state transitions using:
- Rössler attractor fitting
- LSTM-based classification
- Directed Probabilistic Graph modeling

## Files

- `updated_rossler.py`: Class for fitting EEG to chaotic attractor and visualizing alignment.
- `lstm_model.py`: PyTorch LSTM for classifying EEG states.
- `graph_utils.py`: Visualization and construction of probabilistic transition graphs.
- `main_catenet_pipeline.py`: Example runner script.

## Setup

```bash
pip install mne numpy torch matplotlib hyperopt scikit-learn networkx
```

## Usage

```python
python main_catenet_pipeline.py
```

Update paths for EEG `.edf` file and `scaled_rossler.npy`.

## Citation

Please cite the original paper:

> [Your CATE-NET publication details here]
