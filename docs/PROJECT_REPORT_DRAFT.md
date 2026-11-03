# COS30018 Intelligent Systems
## Handwritten Number Recognition System (HNRS)
### Project summary report: working draft, NOT submission ready

**Team:** [Add names and IDs]  
**Date:** [Add submission date]  
**Tutor:** [Add name]

## Contents
1. Introduction
2. System architecture
3. Installation and user guide
4. Data preprocessing
5. Image segmentation
6. Models and training
7. Evaluation protocol
8. Implementation and individual contributions
9. Experiments and analysis
10. Critical analysis
11. Demonstration scenarios
12. Conclusion
13. References

## 1. Introduction

The objective is to recognise multi-digit handwritten numbers using an end-to-end pipeline. The scope includes image acquisition, preprocessing, segmentation, classification and user-facing predictions. The MNIST dataset supports supervised learning for single-digit classifiers.

## 2. System architecture

The Tkinter interface loads an image or composes one from a digit-image folder. OpenCV and NumPy transform the image, identify connected components and produce normalized 28 × 28 patches. TensorFlow/Keras supplies CNN and MLP classifiers. Predictions are concatenated from left to right.

## 3. Installation and user guide

Install Python 3.10–3.12 and dependencies in requirements.txt. Execute python train.py --epochs 5 --model both to train and evaluate both models. Execute python app.py to open the interface. Choose a model, load an image, then click Recognise. The composition tool requires 0.png through 9.png in a chosen folder.

## 4. Data preprocessing

The implementation converts to grayscale, uses border-polarity detection and Otsu thresholding, crops foreground, rescales the longest side to 20 pixels, centres it on a 28 × 28 canvas and normalizes to [0,1]. The MNIST training data uses grayscale normalization but not the exact same crop-and-centre transform. This training/inference mismatch should be investigated experimentally.

## 5. Image segmentation

Connected-component analysis detects isolated foreground components and orders bounding boxes left to right. Its main limitation is touching digits or disconnected digit strokes. Alternative projection profiles, contour-based grouping and watershed segmentation should be compared with measured examples.

## 6. Models and training

CNN: two convolutional layers (32 and 64 filters), pooling, a 128-unit dense layer and dropout. MLP: flattened input, dense layers of 256 and 128 units and dropout. Both use Adam and sparse categorical cross entropy. A 10% validation split is used on MNIST training data. Test-set results are written after training.

## 7. Evaluation protocol

Report the CNN and MLP test-set accuracy, precision, recall, F1, confusion matrices and training/validation curves from the generated results files. Collect an independent handwritten-number set and report digit-level and complete-number accuracy. Test different lighting, spacing, handwriting and image sizes. Record failures without altering test labels.

## 8. Implementation and individual contributions

[REQUIRED: identify each group member, specific code contributions, experimentation, Git commits, weekly worklogs and sprint reports. Do not invent contributions.]

## 9. Experiments and analysis

[REQUIRED: insert actual CNN/MLP metrics, confusion-matrix screenshots, segmentation ablations, number-level evaluation, and observations. No measurements are invented in this draft.]

## 10. Critical analysis

The current connected-components method assumes isolated digits, and the GUI is sensitive to image quality and contrast. Confidence scores are classifier probabilities, not guaranteed calibrated probabilities. Real-world multi-digit accuracy can be substantially lower than MNIST digit accuracy. Further improvements include component merging, deskewing and data augmentation.

## 11. Demonstration scenarios

Demonstrate a clean three-digit input, a five-digit composed image, switching CNN and MLP, a blank image, and an image with touching digits to show limitations. Save screenshots and actual predictions for each scenario.

## 12. Conclusion

The proposed implementation integrates image acquisition, preprocessing, segmentation and classification into a desktop interface. Conclusions about comparative model performance must wait for completed training and evaluation.

## 13. References

[REQUIRED: add verified academic and software references used by the team, with appropriate citations. Suggested starting points: original MNIST research, CNN literature, TensorFlow documentation, OpenCV documentation.]

