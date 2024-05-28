<img src="https://github.com/trtrgfh/Page-Flip-Detection/assets/73056232/0c531186-5c1b-477d-8e78-931298e9c268" width="600"/>

# Page Flip Detection

# Project Overview
This project aims to develop a machine learning model capable of predicting if a page is being flipped using only a single image. The goal is to provide a solution that can automatically detect page flipping in real-time, which can be valuable in various applications such as document scanning, digital book reading, and image processing.

# Installation and Setup
## Python Packages Used
- **General Purpose:** os, pickle
- **Data Manipulation:** numpy, pandas
- **Data Visualization:** matplotlib, PIL
- **Machine Learning:** pytorch, scikit-learn
  
# Data
Dataset used can be found [here](https://drive.google.com/file/d/1KDQBTbo5deKGCdVV_xIujscn5ImxW4dm/view?usp=sharing).

# Results and evaluation
Misclassified examples (true labels vs predicted labels): \
<img src="https://github.com/trtrgfh/Page-Flip-Detection/assets/73056232/a736ef74-527b-40a5-b28b-4b8d6a4d8778" width="700"/>

### CNN
Train_loss: 0.3892, Train_acc: 93.8194, 
Test_loss: 0.3983, Test_acc: 93.1626, Test_f1: 0.9333

### ResNet
Train_loss: 0.3938, Train_acc: 91.8472
Test_loss: 0.3803, Test_acc: 93.1704, Test_f1: 0.9267

### MobileNet
Train_loss: 0.0381, Train_acc: 98.6944
Test_loss: 0.06094, Test_acc: 98.5197, Test_f1: 0.9843

