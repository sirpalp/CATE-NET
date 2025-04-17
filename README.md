[![DOI](https://zenodo.org/badge/DOI/10.1016/j.compbiomed.2025.109832.svg)](https://doi.org/10.1016/j.compbiomed.2025.109832)

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

If you use this toolbox, please cite the original paper:

> Sirpal, P., Sikora, W. A., & Refai, H. H. (2025).  
> *Brain State Network Dynamics in Pediatric Epilepsy: Chaotic Attractor Transition Ensemble Network*.  
> Computers in Biology and Medicine, 188, 109832.  
> https://doi.org/10.1016/j.compbiomed.2025.109832

You can also cite using BibTeX:

<details>
<summary>BibTeX</summary>

```bibtex
@article{sirpal2025catenet,
  title={Brain State Network Dynamics in Pediatric Epilepsy: Chaotic Attractor Transition Ensemble Network},
  author={Sirpal, Parikshat and Sikora, William A and Refai, Hazem H},
  journal={Computers in Biology and Medicine},
  volume={188},
  pages={109832},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.compbiomed.2025.109832}
}
