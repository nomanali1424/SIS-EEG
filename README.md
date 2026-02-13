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
- Interpretability (Feature maps + Grad-CAM)




## Performance

## Performance

<table>
  <tr>
    <th rowspan="2">Data</th>
    <th rowspan="2">Sub. Dep.</th>
    <th rowspan="2">Emotion</th>
    <th rowspan="2">#</th>
    <th colspan="3">Accuracies</th>
  </tr>
  <tr>
    <th>w/o SIS</th>
    <th>SIS</th>
    <th>Improv.</th>
  </tr>

  <!-- ================= DENS ================= -->

  <!-- DENS Dep -->
  <tr>
    <td rowspan="14"><b>DENS</b></td>
    <td rowspan="6">Dep</td>
    <td>V</td><td>2</td><td>97.62</td><td>97.89</td><td>0.27</td>
  </tr>
  <tr><td>V</td><td>3</td><td>96.56</td><td>98.67</td><td>2.11</td></tr>
  <tr><td>A</td><td>2</td><td>97.03</td><td>98.95</td><td>1.92</td></tr>
  <tr><td>A</td><td>3</td><td>97.61</td><td>99.59</td><td>1.98</td></tr>
  <tr><td>V-A</td><td>4</td><td>96.30</td><td>99.02</td><td>2.72</td></tr>
  <tr><td>VAD</td><td>8</td><td>95.70</td><td>98.01</td><td>2.31</td></tr>

  <tr>
    <td colspan="6"><b>Average Improvement: 1.89%</b></td>
  </tr>

  <!-- DENS Indep -->
  <tr>
    <td rowspan="6">Indep</td>
    <td>V</td><td>2</td><td>67.67</td><td>67.69</td><td>0.02</td>
  </tr>
  <tr><td>V</td><td>3</td><td>53.83</td><td>54.44</td><td>0.61</td></tr>
  <tr><td>A</td><td>2</td><td>63.67</td><td>67.98</td><td>4.31</td></tr>
  <tr><td>A</td><td>3</td><td>50.81</td><td>54.22</td><td>3.41</td></tr>
  <tr><td>V-A</td><td>4</td><td>39.47</td><td>40.02</td><td>0.55</td></tr>
  <tr><td>VAD</td><td>8</td><td>22.43</td><td>22.47</td><td>0.04</td></tr>

  <tr>
    <td colspan="6"><b>Average Improvement: 1.49%</b></td>
  </tr>


  <!-- ================= DEAP ================= -->

  <!-- DEAP Dep -->
  <tr>
    <td rowspan="14"><b>DEAP</b></td>
    <td rowspan="6">Dep</td>
    <td>V</td><td>2</td><td>96.30</td><td>97.33</td><td>1.03</td>
  </tr>
  <tr><td>V</td><td>3</td><td>95.10</td><td>97.01</td><td>1.91</td></tr>
  <tr><td>A</td><td>2</td><td>96.31</td><td>97.78</td><td>1.47</td></tr>
  <tr><td>A</td><td>3</td><td>94.58</td><td>96.02</td><td>1.44</td></tr>
  <tr><td>V-A</td><td>4</td><td>94.96</td><td>98.97</td><td>4.01</td></tr>
  <tr><td>VAD</td><td>8</td><td>93.26</td><td>93.88</td><td>0.62</td></tr>

  <tr>
    <td colspan="6"><b>Average Improvement: 1.75%</b></td>
  </tr>

  <!-- DEAP Indep -->
  <tr>
    <td rowspan="6">Indep</td>
    <td>V</td><td>2</td><td>55.67</td><td>55.99</td><td>0.32</td>
  </tr>
  <tr><td>V</td><td>3</td><td>43.18</td><td>44.15</td><td>0.97</td></tr>
  <tr><td>A</td><td>2</td><td>54.55</td><td>54.80</td><td>0.25</td></tr>
  <tr><td>A</td><td>3</td><td>37.81</td><td>38.85</td><td>1.04</td></tr>
  <tr><td>V-A</td><td>4</td><td>29.89</td><td>30.01</td><td>0.12</td></tr>
  <tr><td>VAD</td><td>8</td><td>19.98</td><td>21.00</td><td>1.02</td></tr>

  <tr>
    <td colspan="6"><b>Average Improvement: 0.62%</b></td>
  </tr>

</table>





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
