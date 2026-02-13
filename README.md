<!-- ===================== HERO SECTION ===================== -->

<h1 align="center">
  SIS-EEG: Spatially-Infused Spectrograms for Robust EEG Feature Extraction in Deep Learning Frameworks
</h1>

<p align="center">
  <a href="https://scholar.google.com/citations?user=S6GbuwcAAAAJ&hl=en"><strong>Mohammad Asif</strong></a>,
  <a href="https://nomanali1424.github.io/About/"><strong>Noman Ali</strong></a>,
  <a href="https://www.linkedin.com/in/aditya-gupta-91991215b/"><strong>Aditya Gupta</strong></a>,
  <a href="https://www.linkedin.com/in/diya-srivastava-5a7187160/"><strong>Diya Srivastava</strong></a>,
  <a href="https://scholar.google.com/citations?user=raA5Dc8AAAAJ&hl=en"><strong>Sudhakar Mishra</strong></a>,
  <a href="https://scholar.google.co.in/citations?user=CpLiFv8AAAAJ&hl=en"><strong>Uma Shanker Tiwary</strong></a>
</p>

<p align="center">
  <em>IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2026</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/ICASSP-2026-blue" />
  <img src="https://img.shields.io/badge/Built%20with-TensorFlow-FF6F00?logo=tensorflow" />
  <img src="https://img.shields.io/badge/Domain-EEG-brightgreen" />
  <img src="https://img.shields.io/badge/Status-Accepted-success" />
</p>

<p align="center">
  <!-- Logo placeholder -->
  <img src="assets/sis_logo.png" width="280" alt="SIS-EEG Logo (coming soon)">
</p>

<p align="center">
  <em>
    Official Demo codebase for<br>
    <strong>"Spatially-Infused Spectrograms for Robust EEG Feature Extraction in Deep Learning Frameworks"</strong>
  </em>
</p>

<hr>





# 📌 Overview

This repository provides the official implementation of **SIS-EEG (Spatially-Infused Spectrograms)** — a spatially-aware EEG feature construction framework designed to enhance deep learning performance in affective computing tasks.

The framework supports:

- Multiple datasets: **DENS**, **DEAP**
- Multiple tasks:
  - Arousal (2-class / 3-class)
  - Valence (2-class / 3-class)
  - VAD (8-class)
- Feature variants:
  - **WSIS** (Without Spatial Infusion)
  - **SIS** (Spatially-Infused Spectrograms)
- Automated experiment logging
- Structured checkpoint saving
- Structured result saving
- Interpretability (Feature maps + Grad-CAM)


# 📂 Project Structure

```
SIS-EEG/
│
├── datasets/
│ ├── data_loader.py
│ ├── utils.py
│
├── feature_creation.py
├── model.py
├── main.py
├── interpretability.py
│
├── checkpoints/
├── results/
│
├── environment.yml
├── requirements.txt
└── README.md
```

---

# 📥 Dataset Setup

You must manually download the datasets.

## 🔹 DENS

Place files in:
```
data/DENS/
│
├── Emotional/
│ ├── *.mat
│
└── wholeFrequencyDependentDataWithVADLFR_ReFormattingWholeFrequencyVA.xlsx
```


---

## 🔹 DEAP

Place files in:
```
data/DEAP/
│
├── s01.dat
├── s02.dat
...
└── s32.dat
```

⚠ Folder names must match exactly.

---

# ⚙️ Installation

## 🔹 Option 1 — Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate sis-eeg-env

```
## 🔹 Option 2 — Pip 
```bash
pip install -r requirements.txt
```

---
# 🚀 Training

Run experiments using:

```bash
python main.py --dataset_name DATASET --task TASK --num_classes N --feature_type FEATURE
```

🔹 Available Arguments
Argument	Options	Description
--dataset_name	DENS, DEAP	Dataset selection
--task	A, V, VAD	Classification task
--num_classes	2, 3, 8	Number of output classes
--feature_type	SIS, WSIS	Feature construction type

🔹 Example Commands
Arousal (2-class) with SIS on DENS

```bash
python main.py --dataset_name DENS --task A --num_classes 2 --feature_type SIS
```

# 🔬 Interpretability

To generate feature maps and Grad-CAM:
```bash
python interpretability.py --dataset_name DENS --task A --num_classes 2 --feature_type SIS
```