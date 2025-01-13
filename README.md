# **DistalRadiusPrediction**

This deep learning project focuses on the **prediction of distal radius fracture redisplacement** using pre-reduction and post-reduction radiographs along with patient metadata. The goal is to develop a predictive model that can assist clinicians in determining whether to **recommend surgery** to prevent the fracture from redisplacing into **unacceptable alignment**.

---

## **Background**  
Distal radius fractures are among the most common orthopedic injuries. While many of these fractures can be treated conservatively, some may redisplace after initial reduction, leading to poor functional outcomes if left untreated. Predicting the risk of redisplacement can help guide clinical decisions on whether to recommend surgical intervention. However, predicting redisplacement is inherently uncertain due to variability in fracture patterns, patient factors, and healing processes. A binary prediction (i.e., redisplacement or no redisplacement) may lead to **overconfident decisions** that do not fully capture this uncertainty. To address this, **calibrated probabilities** are preferred, as they provide a measure of confidence that better reflects the true risk of redisplacement, making the prediction more useful in clinical decision-making.

In this project, a **two-step  approach** was used:
1. **Caibrated risk probabilities prediction using XGBoost:**  
   Radiographic measurements obtained from pre-reduction and post-reduction radiographs were used as input features for an **XGBoost** model to predict the risk of fracture redisplacement.
2. **Deep Learning Model Training with Calibrated Risk as a Soft Target:**  
   The calibrated risk of redisplacement from the XGBoost model was used as a **soft target** for training the deep learning model. By predicting the **calibrated probability** of redisplacement rather than a binary outcome (redisplacement vs. no redisplacement), leading to better performance compared to training on binary labels alone.

After training, the deep learning model can predict the risk of redisplacement **directly from radiographs and patient metadata**, eliminating the need for manual radiographic measurements. This reduces clinician workload and variability while providing an end-to-end, automated prediction.

---

## **Deep Learning Model Architecture**  
The deep learning model was designed with a **customized multi-image architecture** that takes in multiple radiographic images (pre-reduction and post-reduction) along with patient metadata (e.g., age, sex, etc.). The following pre-trained backbones from `timm` were used for feature extraction:
- **CoAtNet_0:** A hybrid convolution-transformer backbone for capturing both local and global image features.
- **EfficientNetV2_S:** A CNN-based backbone known for its high accuracy and efficient performance.

By combining radiographic features and clinical data, the model leverages the calibrated risk as an informative target, improving its ability to predict redisplacement while incorporating both **radiographic patterns** and **clinical context**.

---

## **Purpose**  
The ultimate goal of this project is to create a predictive tool that supports clinicians in making **data-driven decisions** about the need for surgical fixation, thereby improving patient outcomes and minimizing unnecessary surgeries for distal radius fractures.

---

## **Dependencies**
This project requires the following Python libraries:
- `torch`: For deep learning model development ([PyTorch Official Website](https://pytorch.org/))
- `timm`: For pre-trained backbone models (CoAtNet, EfficientNetV2) ([GitHub Repository](https://github.com/rwightman/pytorch-image-models))
- `pydicom`: For DICOM image handling ([PyDICOM Documentation](https://pydicom.github.io/))
- `scikit-learn`: For building the XGBoost model and calculating metrics ([Scikit-learn Documentation](https://scikit-learn.org/stable/))
- `xgboost`: For the risk prediction model ([XGBoost Documentation](https://xgboost.readthedocs.io/))
- `numpy`: For data handling ([NumPy Documentation](https://numpy.org/))
- `matplotlib`: For data visualizations ([Matplotlib Documentation](https://matplotlib.org/))
- `pandas`: For handling patient metadata ([Pandas Documentation](https://pandas.pydata.org/))
- `cv2 (OpenCV)`: For image preprocessing and computer vision tasks ([OpenCV Documentation](https://opencv.org/))
