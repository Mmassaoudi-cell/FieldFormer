# FieldFormer

Source code for a compact, joint representation-learning architecture for industrial control network threat analysis. FieldFormer replaces a large generative packet-language-model pretraining stage with a masked-token packet encoder combined with a lightweight selective state-space (SSM) temporal backbone, trained jointly for multiclass attack classification and semisupervised anomaly detection on ICS/IIoT network traffic (Modbus/TCP process-automation traffic, DNP3, and IIoT protocols).

## Repository structure

```
FieldFormer/
├── configs/
│   └── final_model_config.yaml     # frozen hyperparameters of the final model
├── src/
│   ├── dissect.py                  # protocol-aware packet field dissection
│   ├── find_onset.py               # attack-onset detection for window labeling
│   ├── build_modbus_dataset.py     # Modbus/TCP PCAP -> packet/window parquet
│   ├── build_dnp3_features.py      # DNP3 flow-feature extraction
│   ├── build_edgeiiotset_dataset.py# Edge-IIoTset PCAP -> packet/window parquet
│   ├── add_more_benign.py          # benign-class balancing
│   ├── train_tokenizer.py          # shared BPE + hex-byte tokenizer
│   ├── train_fieldformer_encoder.py# masked-token packet encoder pretraining
│   ├── ssm_layer.py                # selective state-space temporal layer
│   ├── train_candidate_a.py .. train_candidate_d.py  # architecture candidates
│   ├── train_final_model.py        # final frozen-config training run
│   ├── tune_final_model.py         # validation-only Optuna hyperparameter search
│   ├── extract_embeddings.py, extract_fieldformer_embeddings.py
│   ├── train_bart_teacher.py, train_bart_baseline.py, train_distillation.py
│   ├── train_e2e_transformer_baseline.py  # independent end-to-end baseline
│   ├── train_classical_benchmarks.py, train_deep_benchmarks.py,
│   │   train_anomaly_benchmarks.py, train_dnp3_benchmarks.py, train_lstm_downstream.py
│   ├── eval_heldout_class.py       # held-out attack-class generalization
│   ├── eval_cross_dataset_edgeiiotset.py  # zero-shot cross-dataset transfer
│   ├── run_heldout_multiseed.py, run_ablations.py, run_ablations_heldout.py,
│   │   run_robustness.py
│   ├── measure_efficiency.py       # latency / parameter-count benchmarking
│   └── statistical_tests.py        # Wilcoxon signed-rank + Holm correction
├── requirements.txt
└── LICENSE
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.11+ and a CUDA-capable GPU are recommended for training.

## Dataset download instructions

Datasets are not committed to this repository. Download them separately and place them under a local `external_datasets/` directory (or point the scripts' `DATA_ROOT` constants at your own location):

- **Frazão et al. (2019) Modbus/TCP process-automation PCAPs** — public, CC-BY-3.0, University of Coimbra / ATENA H2020: https://github.com/tjcruz-dei/ICS_PCAPS (release `MODBUSTCP#1`). Expected layout: `clean/`, `mitm/`, `modbusQueryFlooding/`, `modbusQuery2Flooding/`, `pingFloodDDoS/`, `tcpSYNFloodDDoS/`.
- **DNP3 intrusion-detection dataset** (Radoglou-Grammatikis et al., 2022) — public, CC-BY-4.0, Zenodo mirror of the IEEE Dataport original: https://zenodo.org/record/7348493. Expected layout: one folder per attack scenario with raw DNP3 PCAPs and labeled flow CSVs.
- **Edge-IIoTset** (Ferrag et al., *IEEE Access*, 2022) — public, available via IEEE Dataport and a Kaggle mirror under the title "Edge-IIoTset Cyber Security Dataset of IoT & IIoT". Used here only as a held-out, zero-shot cross-dataset evaluation set.

## How to train

```bash
# 1. Build the datasets (packet + window level parquet files)
python src/build_modbus_dataset.py
python src/build_dnp3_features.py
python src/build_edgeiiotset_dataset.py

# 2. Train the shared tokenizer and the masked-token packet encoder
python src/train_tokenizer.py
python src/train_fieldformer_encoder.py

# 3. Train the final model (frozen configuration, see configs/final_model_config.yaml)
python src/train_final_model.py
```

## How to evaluate

```bash
python src/eval_heldout_class.py --seed 0 --confirmatory   # held-out attack-class generalization
python src/eval_cross_dataset_edgeiiotset.py                # zero-shot cross-dataset transfer
python src/measure_efficiency.py                            # latency / parameter count
python src/statistical_tests.py                              # significance testing across benchmarks
```

## How to reproduce the paper's experiments

1. Run `train_classical_benchmarks.py`, `train_deep_benchmarks.py`, and `train_anomaly_benchmarks.py` for the full baseline suite.
2. Run `train_candidate_a.py` through `train_candidate_d.py` to reproduce the architecture-candidate screening.
3. Run `run_heldout_multiseed.py` for the multi-seed held-out generalization study.
4. Run `run_ablations.py` and `run_ablations_heldout.py` for the ablation study.
5. Run `run_robustness.py` for the robustness evaluation.
6. Run `train_e2e_transformer_baseline.py` for the independent (non-pretrained) benchmark used to isolate representation-sharing confounds.

Each script writes its raw results to a local `results/` directory as JSON/CSV; no results are precomputed or bundled in this repository.

## Citation

```bibtex
@article{massaoudi_fieldformer,
  author  = {Massaoudi, Mohamed and Ez Eddin, Maymouna},
  title   = {Joint Representation Learning for Industrial Control Network Threat Analysis: A Compact State-Space Alternative to Generative Packet-Language-Model Pretraining},
  journal = {IEEE Transactions (under review)},
  year    = {2026}
}
```

## License

MIT — see `LICENSE`.

## Contact

Mohamed Massaoudi — mohamed.massaoudi@tamu.edu
