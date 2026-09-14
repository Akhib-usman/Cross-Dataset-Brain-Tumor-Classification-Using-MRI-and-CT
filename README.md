# Cross-Dataset Brain Tumor Detection Using Independent MRI and CT Models

1. As my deep learning project, I tried to build a brain tumor classification model using MRI and CT images.
2. The difference between this and other basic projects is I'm trying to classify using 2 type of image modalities(MRI, CT).
3. The project trains separate models for each imaging modality(MRI, CT) and evaluates them on a different dataset to study cross-dataset performance or u can say it as Generalization.

An Imp Note for readers: MRI and CT are trained using separate models. They are combined only at the prediction level and are not jointly trained as a single multimodal model.

## Project Overview

The project uses two independent deep learning models:

- **MRI Model** – trained on MRI images
- **CT Model** – trained on CT images

Both models classify images into:

- Healthy
- Tumor

The models are trained using **Dataset-1** and tested on **Dataset-2**.

For the final prediction, an MRI scan and a CT scan can be provided separately.
Their tumour probabilities are then combined using **decision-level fusion** to
produce a final prediction.

---

## Project Workflow

```text
                    Dataset-1
                       │
             ┌─────────┴─────────┐
             │                   │
            MRI                  CT
             │                   │
             ▼                   ▼
       MRI Model             CT Model
       EfficientNetB0        EfficientNetB0
             │                   │
             └─────────┬─────────┘
                       │
                 Dataset-2 Testing
                       │
             ┌─────────┴─────────┐
             │                   │
        MRI Prediction      CT Prediction
             │                   │
             └─────────┬─────────┘
                       │
                Final Prediction
              (Decision-Level Fusion)
```
---

## Datasets

The datasets are approx 900 MB and 400MB so due to size constraints I can't directly pull them into repo.\
So I'm sharing them as ZIP files through Google Drive links.

**Dataset-1**
  - Dataset-1 is used for training and validation. ~(900 MB)
```text
      Dataset-1
      │
      ├── MRI
      │   ├── Healthy
      │   └── Tumor
      │
      └── CT
          ├── Healthy
          └── Tumor
```
Download: https://drive.google.com/file/d/1JuWrSAKTsX8tvPrpJUOQvpYUXUiM9E1K/view?usp=sharing

**Dataset-2**
  - Dataset-2 is used only for cross-dataset testing. ~(400MB)
```text
Dataset-2
│
├── MRI
│   ├── Healthy
│   └── Tumor
│
└── CT
    ├── Healthy
    └── Tumor
```
Download: https://drive.google.com/file/d/1pBWU65hgU-seXJupjdpDAdkWHZU9XJCp/view?usp=sharing

An Imp Note for readers: Extract both datasets into the project folder and keep the folder structure and names exactly as shown above. **Dont change any names**.

## Dataset Statistics:
**Dataset-1**
| Modality | Healthy | Tumor |  Total |
| -------- | ------: | ----: | -----: |
| MRI      |   5,600 | 5,900 | 11,500 |
| CT       |   5,620 | 5,662 | 11,282 |

**Dataset-2**
| Modality | Healthy | Tumor | Total |
| -------- | ------: | ----: | ----: |
| MRI      |   2,000 | 3,000 | 5,000 |
| CT       |   2,300 | 2,318 | 4,618 |


## Model
  - Both models use EfficientNetB0 with ImageNet transfer learning.
  - Images are resized to 224 × 224 and converted to RGB.

**Model Architecture**
```text
Input Image
     ↓
EfficientNetB0
     ↓
Global Average Pooling
     ↓
Dense Layer
     ↓
Sigmoid
     ↓
Tumor Probability
```
**Training & Testing**
```text
Dataset-1
    │
    ├── 80% → Training
    │
    └── 20% → Validation
             │
             ▼
       Trained Models
             │
             ▼
        Dataset-2
             │
             ▼
       Cross-Dataset
          Testing
```
The models are evaluated using:
1) Accuracy
2) Precision
3) Recall
4) F1 Score
5) AUC
6) Confusion Matrix

## Final Prediction
In final cell in the .ipynb it asks the user for a Mri and a Ct scan:
```text
MRI Scan
    ↓
MRI Model
    ↓
MRI Tumor Probability
          \
           → Decision-Level Fusion → Final Prediction
          /
CT Tumor Probability
    ↑
CT Model
    ↑
CT Scan
```
As of now i'm current using equal weights.
  - MRI Weight = 0.5
  - CT Weight  = 0.5
```text
Fused Probability = (0.5 × MRI Probability) + (0.5 × CT Probability)
```
The final output displays the MRI probability, CT probability, fused probability, and final prediction.

---

# How to Run
**1. Clone the repository**

**2. Download the datasets and extract**
  - Dataset-1.zip
  - Dataset-2.zip\
from the Google Drive links above.

**3. Open the notebook and run the cells in order**

**Requirements**
  - Python 3.12
  - TensorFlow 2.18.1
  - TensorFlow Metal
  - NumPy
  - Pandas
  - Matplotlib
  - Seaborn
  - Scikit-learn
  - Pillow
  - Jupyter Notebook

## Disclaimer
**I built This project only for academic and research purposes.**
**The model predictions should not be considered a medical diagnosis or a replacement for evaluation by a qualified medical professional.**
