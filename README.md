# OFM-U-Net
Optical Flow-Enhanced Mamba U-Net for Cardiac Phase Detection in Ultrasound Videos

## 💡 Motivation
The detection of cardiac phase in ultrasound videos, identifying end-systolic (ES) and end-diastolic (ED) frames, is a critical step in assessing cardiac function, monitoring structural changes, and diagnosing congenital heart disease.
Current popular methods use recurrent neural networks to track dependencies over long sequences, but overlook the short-term movement of heart valves that sonographers normally rely on.
In this research, we propose a novel optical flow-enhanced Mamba U-net framework, designed to utilize both short-term motion and long-term information to detect the cardiac cycle phase in ultrasound videos.

## 🛠️ Usage

### 1. Environment Setup
Clone the repository and set up the required environment.

### 2. Data Preparation
#### Fetal Echocardiogram Dataset
Each video in our dataset corresponds to an individual JSON file. The structure of the JSON file is outlined below:

```json
{
  "annotations": {
    "0": 0.0,
    "1": 0.6933612743506347,
    "2": 0.8735804647362989,
    "3": 1.0,
    "4": 0.7023319615912207,
    "5": 0.4705075445816187,
    "6": 0.2962962962962962,
  }
}
```
- **`annotations`**: This is a dictionary where each key represents a frame index of the video, and the value indicates an annotation value, with ES targeting 0 and ED targeting 1.
#### Adult Echonet-Dynamic Dataset
Ensure to download and prepare the [Echonet-Dynamic](https://echonet.github.io/dynamic/) dataset, ensuring the directory structure matches the specified requirements.

### 3. Training & Evaluation
To train a model, you can simply run  (run/run_training.py)
```bash
python run_training.py 
```
To test the trained model, you can simply run (inference/predict_from_raw_data.py)
```bash
python predict_from_raw_data.py
```

# Best Performing Video Prediction

![Converted GIF](opticalflowvideo/bestpre.gif)



# Worst Performing Video Prediction

![Converted GIF](opticalflowvideo/worstpre.gif)

