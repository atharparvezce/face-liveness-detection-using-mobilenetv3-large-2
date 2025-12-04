# Face Liveness Detection using MobileNetV3-Large

A deep-learning based face anti-spoofing system using
**MobileNetV3-Large**, trained on a video-based dataset and evaluated
using industry-standard liveness metrics (FAR, FRR, HTER).\
This repository includes full training pipeline, evaluation scripts,
Grad-CAM visualizations, and t-SNE embedding analysis.

------------------------------------------------------------------------

## 🔥 Key Features

-   MobileNetV3-Large backbone\
-   Custom Train/Validation/Test split\
-   Random frame sampling per video for training\
-   First-frame evaluation for validation & testing\
-   Metrics: Accuracy, FAR, FRR, HTER\
-   Automatic best-model checkpoint saving\
-   Grad-CAM visualization\
-   t-SNE embedding analysis

------------------------------------------------------------------------

## 📊 Dataset Split

  Split        Samples
  ------------ ---------
  Train        1391
  Validation   350
  Test         1748
  Total        3497

------------------------------------------------------------------------

## 🏗 Model Architecture

-   Backbone: MobileNetV3-Large\
-   Input: 224×224\
-   Loss: BCEWithLogitsLoss\
-   Optimizer: Adam\
-   Scheduler: ReduceLROnPlateau\
-   Output: 1 logit (real vs attack)

------------------------------------------------------------------------

## 📈 Evaluation Metrics

-   Accuracy\
-   FAR\
-   FRR\
-   HTER = (FAR + FRR) / 2

------------------------------------------------------------------------

## 🧪 Example Results

    Test Loss : 0.0403
    Test Acc  : 0.9863
    FAR       : 0.0000
    FRR       : 0.0535
    HTER      : 0.0267

------------------------------------------------------------------------

## 🔍 Grad-CAM

Used to highlight important facial regions influencing model decisions.

------------------------------------------------------------------------

## 🎨 t-SNE

Visualizes feature-space separation between real and attack samples.

------------------------------------------------------------------------

## 🛠 Requirements

    torch
    torchvision
    numpy
    sklearn
    matplotlib
    opencv-python
    tqdm
    pandas

------------------------------------------------------------------------

## 📜 License

MIT License
