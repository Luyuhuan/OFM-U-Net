# OFM-U-Net
Optical Flow-Enhanced Mamba U-Net for Cardiac Phase Detection in Ultrasound Videos

## Motivation
The detection of cardiac phase in ultrasound videos, identifying end-systolic (ES) and end-diastolic (ED) frames, is a critical step in assessing cardiac function, monitoring structural changes, and diagnosing congenital heart disease.
Current popular methods use recurrent neural networks to track dependencies over long sequences, but overlook the short-term movement of heart valves that sonographers normally rely on.
In this research, we propose a novel optical flow-enhanced Mamba U-net framework, designed to utilize both short-term motion and long-term information to detect the cardiac cycle phase in ultrasound videos.

## 🛠️ Usage
### ⚠️ Important
This is a preliminary version of the code and may contain errors. The correct version will be released shortly. We apologize for any inconvenience.

### 1. Environment Setup
Clone the repository and set up the required environment.

### 2. Data Preparation & Pre-trained Weights
#### Pascal VOC 2012 Dataset
Download the dataset and unzip it. Detailed instructions are included for data setup.

#### Cityscapes and COCO Datasets
Ensure to download and prepare Cityscapes and COCO datasets, ensuring the directory structure matches the specified requirements.

### 3. Training & Evaluation
To train and evaluate the model, follow the provided scripts that are adapted for multiple GPU setups.

### 4. Results
Model performance and logs will be updated soon. Specific sections will detail the outcomes for different datasets.

## Citation
If our work assists your research, please consider citing:
```bibtex
@inproceedings{allspark2024,
  title={AllSpark: Reborn Labeled Features from Unlabeled in Transformer for Semi-Supervised Semantic Segmentation},
  author={Wang, Haonan and Zhang, Qixiang and Li, Yi and Li, Xiaomeng},
  booktitle={CVPR},
  year={2024}
}
