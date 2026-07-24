# TrustOCTAI — Trustworthy Retinal OCT Disease Classification

Official PyTorch Research Implementation of **TrustOCTAI**: An Evidence-Based, Domain-Generalizable Framework for Retinal OCT Disease Classification.

---

## 🌟 Key Features

- **Multi-Scale Feature Fusion (MSF)**: Fuses mid-level spatial details ($x_3$, 1024-ch) with deep semantic representations ($x_4$, 2048-ch) to capture multi-scale retinal lesions (drusen to large fluid pockets).
- **CBAM Dual Attention**: Channel & Spatial attention modules focus representation on the Retinal Pigment Epithelium (RPE) layer while suppressing ocular background noise.
- **Mid-Level MixStyle Domain Generalization**: Perturbs mid-level feature statistics ($x_3$) to randomize scanner-specific contrast/speckle noise across Heidelberg, Topcon, and Cirrus devices.
- **Evidential Dirichlet Head**: Replaces standard Softmax with Subjective Logic Dirichlet distributions ($\alpha_k = e_k + 1$) for single-forward-pass epistemic & aleatoric uncertainty quantification.
- **Selective Classification & Referral**: Enables uncertainty-aware clinical triage, achieving **>99% accuracy** on non-referred diagnostic scans.
- **Zero-Shot External Validation**: Tested across cross-hospital OCTID datasets for robust out-of-distribution evaluation.

---

## 📁 Repository Structure

```text
TrusthOCTAI/
│
├── README.md
├── requirements.txt
├── main.py
│
├── configs/
│   ├── dataset.yaml
│   ├── model.yaml
│   └── train.yaml
│
├── datasets/
│   ├── dataset.py
│   ├── transforms.py
│   └── utils.py
│
├── models/
│   ├── resnet50.py
│   ├── msf.py
│   ├── cbam.py
│   ├── mixstyle.py
│   ├── edl_head.py
│   └── trustoct.py
│
├── engine/
│   ├── trainer.py
│   ├── validator.py
│   └── tester.py
│
├── evaluation/
│   ├── metrics.py
│   ├── calibration.py
│   ├── robustness.py
│   └── report.py
│
├── explainability/
│   ├── layercam.py
│   └── visualization.py
│
└── notebooks/
    └── TrustOCT_Colab.ipynb
```

---

## 🚀 Quick Start (Google Colab)

1. Open `notebooks/TrustOCT_Colab.ipynb` in Google Colab.
2. Clone repository:
   ```bash
   !git clone https://github.com/Gnanapravallika/TrusthOCTAI.git
   %cd TrusthOCTAI
   ```
3. Run training:
   ```bash
   python main.py
   ```

---

## 📄 Citation & License
Developed for publication in top-tier medical AI journals (IEEE TMI / MedIA). Distributed under the MIT License.
