[![DOI](https://zenodo.org/badge/DOI/10.1016/j.compbiomed.2025.109832.svg)](https://doi.org/10.1016/j.compbiomed.2025.109832)

# CATE-NET: Chaotic Attractor Transition Ensemble Network

**CATE-NET** is an open-source Python toolbox for modeling nonlinear brain state transitions using chaotic attractor theory, deep sequence learning, and directed probabilistic graphs. Originally developed for pediatric epilepsy EEG, it enables:

- Alignment of EEG segments to chaotic attractors (Rössler systems)
- Trajectory-based embedding of neural dynamics
- Brain state decoding via LSTM models
- Visualization of latent transitions via directed probabilistic graphs (DPGM)

## Features

- Chaotic attractor fitting of EEG using Rössler systems
- Low-dimensional trajectory extraction and alignment
- Deep LSTM-based brain state classifier
- Directed probabilistic graph modeling of transitions
- Modular and reusable codebase for neurodynamics research

## Files

| File | Description |
|------|-------------|
| `updated_rossler.py` | Simulates Rössler attractor and fits EEG trajectories |
| `generate_scaled_rossler.py` | Scales and saves chaotic attractor segments |
| `lstm_model.py` | Defines PyTorch-based LSTM classifier |
| `graph_utils.py` | Constructs and visualizes directed probabilistic graphs |
| `main_catenet_pipeline.py` | Main pipeline to run attractor → LSTM → DPGM |
| `load_chbmit_segment.py` | Preprocesses EEG segments from CHB-MIT format |
| `LICENSE` | MIT License for reuse and distribution |
| `CITATION.cff` | GitHub-compatible citation metadata |
| `CATENET_Package.zip` | Archive of the full project |

## Installation

```bash
pip install mne numpy torch matplotlib hyperopt scikit-learn networkx
```
## Usage
To run the full CATE-NET pipeline on a sample EEG segment:

`python main_catenet_pipeline.py`

## Dataset
This software was developed and validated on the CHB-MIT Scalp EEG Database:

EEG recordings from 23 pediatric epilepsy patients (ages 10–22)

Over 900 hours of scalp EEG

173 documented seizure events (tonic, clonic, atonic)

Ideal for testing seizure detection algorithms in clinically realistic scenarios

## Citation
If you use this toolbox, please cite the original paper:

Sirpal, P., Sikora, W. A., & Refai, H. H. (2025).
Brain State Network Dynamics in Pediatric Epilepsy: Chaotic Attractor Transition Ensemble Network.
Computers in Biology and Medicine, 188, 109832.
https://doi.org/10.1016/j.compbiomed.2025.109832

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
```
## License
MIT License — see the LICENSE file for details.
