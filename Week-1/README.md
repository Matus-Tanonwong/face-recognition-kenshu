# Week 1 — Introduction to Face Recognition with PCA and Eigenfaces

## Overview

In Week 1, students learn the basic workflow of face recognition, starting from face detection and progressing to classical face recognition using **Principal Component Analysis (PCA)** and **Eigenfaces**.

The goal is to understand how face images can be represented numerically, reduced to a lower-dimensional feature space, and compared for identity recognition.

This week also introduces a simple webcam-based face enrollment and recognition system.

---

## Learning Objectives

By the end of Week 1, students should be able to:

- Understand the basic pipeline of a face recognition system.
- Understand the difference between **face detection** and **face recognition**.
- Detect faces using a Haar Cascade classifier.
- Understand how a grayscale face image is represented as numerical data.
- Understand the basic idea of Principal Component Analysis (PCA).
- Understand mean subtraction, dimensionality reduction, and Eigenfaces.
- Project face images into PCA feature space.
- Reconstruct face images using a limited number of principal components.
- Perform simple face recognition using nearest-neighbor matching.
- Evaluate recognition performance using accuracy and a confusion matrix.
- Understand the difference between registered and unregistered identities.
- Run a simple webcam-based Eigenfaces recognition demonstration.

---

## Week 1 Materials

```text
Week-1/
│
├── Anaconda_VSCode_GitHub_Windows11_Bilingual_Guide.pptx
├── Haar_Cascade_Face_Detection_Bilingual.pptx
├── Eigenfaces_PCA_Bilingual_Student.ipynb
├── webcam_eigenfaces.py
├── requirements.txt
├── Yale_Faces_Images/
│   ├── README_EN_JA.txt
│   ├── labels.csv
│   ├── person_01/
│   ├── person_02/
│   ├── ...
│   └── person_38/
└── saved_model/
    ├── eigenfaces.joblib
    └── gallery.npz
```

---

## 1. Environment Setup

The following material explains how to prepare the programming environment:

```text
Anaconda_VSCode_GitHub_Windows11_Bilingual_Guide.pptx
```

Students should prepare:

- Python
- Anaconda / Conda
- Visual Studio Code
- Jupyter Notebook
- Git
- GitHub

### Recommended Python Environment

Create a new Conda environment:

```bash
conda create -n face-recognition python=3.11
```

Activate it:

```bash
conda activate face-recognition
```

Install the required packages:

```bash
pip install -r requirements.txt
```

The main packages used in Week 1 are:

```text
numpy
opencv-python
scikit-learn
joblib
```

The Jupyter notebook may additionally require:

```bash
pip install matplotlib pandas pillow jupyter
```

---

## 2. Face Detection

Before recognizing a person, the system must first locate the face in an image.

The Week 1 introduction to face detection is provided in:

```text
Haar_Cascade_Face_Detection_Bilingual.pptx
```

Students learn the basic concept of the **Haar Cascade classifier**, which is a classical method for detecting frontal faces.

The important distinction is:

```text
Face Detection
      ↓
Where is the face?

Face Recognition
      ↓
Whose face is it?
```

Face detection is therefore usually one of the preprocessing steps of a face recognition system.

---

## 3. PCA and Eigenfaces

The main practical exercise for Week 1 is:

```text
Eigenfaces_PCA_Bilingual_Student.ipynb
```

The notebook introduces face recognition step by step.

The general pipeline is:

```text
Face Image
    ↓
Grayscale Image
    ↓
Resize
    ↓
Flatten into a Vector
    ↓
Subtract Mean Face
    ↓
PCA / SVD
    ↓
Eigenface Features
    ↓
Nearest-Neighbor Matching
    ↓
Predicted Identity
```

---

## 3.1 Face Images as Numerical Data

Each face image can be represented as a matrix of pixel values.

The original Yale images have a resolution of approximately:

```text
168 × 192 pixels
```

For faster PCA computation, the notebook resizes the images to:

```text
42 × 48 pixels
```

Therefore, each face becomes a vector containing:

```text
42 × 48 = 2016 values
```

A face image can therefore be considered a point in a **2016-dimensional space**.

---

## 4. Mean Face

The first important PCA preprocessing step is to calculate the average face.

For training samples \(x_i\), the mean face is:

\[
\mu = \frac{1}{N}\sum_{i=1}^{N}x_i
\]

Each face is then centered by subtracting the mean:

\[
x_i^{(c)} = x_i - \mu
\]

This allows PCA to focus on variations between faces rather than the overall average appearance.

---

## 5. Principal Component Analysis

PCA finds directions in the data where the variation is largest.

These directions are called **principal components**.

For face images, the principal components can be reshaped back into images. These images are called **Eigenfaces**.

An Eigenface is not the face of a particular person. Instead, it represents a pattern of variation that appears across many face images.

Examples include variation caused by:

- facial structure
- lighting
- shadows
- expression
- orientation

---

## 6. Singular Value Decomposition

The Eigenfaces method can be explained using Singular Value Decomposition:

\[
X_c = U\Sigma V^T
\]

where:

- \(X_c\) is the centered face data matrix,
- \(U\) contains important directions in image space,
- \(\Sigma\) contains singular values,
- \(V\) describes the corresponding sample-space directions.

In the notebook, PCA is implemented using `scikit-learn`.

---

## 7. Dimensionality Reduction

Instead of comparing all 2016 pixel values directly, PCA represents each face using a smaller number of features.

For example:

```text
Original image:
2016 values

        ↓ PCA

Reduced representation:
80 values
```

The PCA feature representation is:

\[
z = U_k^T(x-\mu)
\]

where \(k\) is the number of retained principal components.

The vector \(z\) is the PCA feature vector used for recognition.

---

## 8. Face Reconstruction

A compressed PCA representation can also be transformed back into an approximate face image:

\[
\hat{x} = \mu + U_kz
\]

Students can experiment with different numbers of principal components, for example:

```text
2 components
10 components
20 components
40 components
80 components
```

Using more components generally preserves more visual information.

However:

> Better reconstruction does not necessarily mean better face recognition.

---

## 9. Face Recognition with Nearest Neighbor

After projecting faces into PCA space, a simple recognition method can compare a test face with registered faces.

The closest registered face is found using Euclidean distance:

\[
i^* = \arg\min_i \|z_{\text{test}}-z_i\|_2
\]

The identity associated with the nearest feature vector is returned as the prediction.

This method is called **1-Nearest Neighbor (1-NN)**.

---

## 10. Training and Testing

The notebook uses a stratified train/test split.

Approximately:

```text
80% → Training
20% → Testing
```

PCA must be learned using **training images only**.

The test images must not be used when calculating:

- the mean face
- the PCA components
- the Eigenfaces

Otherwise, information from the test set would leak into the model.

This problem is called **data leakage**.

---

## 11. Recognition Evaluation

Students evaluate the face recognition system using metrics such as:

- Recognition accuracy
- Confusion matrix
- Correct recognition examples
- Incorrect recognition examples

The notebook also investigates how the number of PCA components affects recognition accuracy.

---

## 12. Unknown / Unregistered Faces

A normal nearest-neighbor classifier always returns one of the registered identities.

For example:

```text
Unknown Person
      ↓
PCA
      ↓
Nearest Neighbor
      ↓
person_07
```

This does not necessarily mean that the unknown person is actually `person_07`.

Week 1 therefore also introduces the idea of **open-set recognition**.

A simple rejection rule can be written as:

\[
\hat{y} =
\begin{cases}
y_{i^*}, & d_{\min} \leq \tau \\
\text{Unknown}, & d_{\min} > \tau
\end{cases}
\]

where \(\tau\) is a distance threshold.

This simple threshold is included only as an educational demonstration and should not be considered a reliable authentication method.

---

## 13. Yale Face Dataset

The dataset used in this exercise is stored in:

```text
Yale_Faces_Images/
```

It contains:

- **38 people**
- approximately **2410 grayscale face images**
- multiple illumination conditions

Example structure:

```text
Yale_Faces_Images/
├── person_01/
│   ├── face_01.png
│   ├── face_02.png
│   └── ...
├── person_02/
│   ├── face_01.png
│   ├── face_02.png
│   └── ...
└── person_38/
```

The folder names are used only as identity labels. They do not represent the real identities of the people in the dataset.

More information is available in:

```text
Yale_Faces_Images/README_EN_JA.txt
```

---

## 14. Running the Jupyter Notebook

Start Jupyter:

```bash
jupyter notebook
```

Then open:

```text
Eigenfaces_PCA_Bilingual_Student.ipynb
```

Run the cells sequentially from the beginning.

The notebook covers:

1. Loading face images
2. Image vectorization
3. Train/test splitting
4. Mean face calculation
5. PCA
6. Eigenface visualization
7. Explained variance
8. PCA projection
9. Face reconstruction
10. Reconstruction error
11. PCA feature visualization
12. Nearest-neighbor recognition
13. Recognition accuracy
14. Confusion matrix
15. Unknown-person experiments
16. Student exercises

---

## 15. Webcam Eigenfaces Demo

The file:

```text
webcam_eigenfaces.py
```

provides a simple webcam-based demonstration.

It combines:

```text
Webcam
   ↓
Haar Cascade Face Detection
   ↓
Face Crop
   ↓
Grayscale + Resize
   ↓
PCA
   ↓
Nearest-Neighbor Matching
   ↓
Identity Prediction
```

### Step 1 — Train the PCA Model

Run:

```bash
python webcam_eigenfaces.py train
```

The program learns the PCA space using the Yale face dataset.

By default, it uses up to:

```text
80 principal components
```

The trained model is stored in:

```text
saved_model/eigenfaces.joblib
```

and the feature gallery is stored in:

```text
saved_model/gallery.npz
```

### Step 2 — Register a New Person

For example:

```bash
python webcam_eigenfaces.py enroll --id student_01
```

The webcam will capture multiple face images of the student.

The enrollment procedure includes several face conditions, such as:

- looking straight
- looking slightly left
- looking slightly right
- smiling
- looking slightly upward

The goal is to provide several samples of the same person.

### Step 3 — Check Registered Identities

Run:

```bash
python webcam_eigenfaces.py status
```

The program displays the identities currently stored in the gallery.

### Step 4 — Run Live Recognition

Run:

```bash
python webcam_eigenfaces.py recognize
```

Press `Q` or `ESC` to quit.

---

## 16. Student Exercises

### Exercise A — Image Resolution

Change the PCA image resolution.

For example:

```python
H = 32
W = 28
```

Observe:

- computation time
- reconstruction quality
- recognition accuracy

### Exercise B — Number of Neighbors

Compare:

```text
1-NN
```

with:

```text
3-NN
```

Does using more neighbors improve recognition accuracy?

### Exercise C — Number of Eigenfaces

Reconstruct a face using only:

```text
2 Eigenfaces
```

Then compare it with reconstructions using more components.

Explain the visible differences.

### Exercise D — Validation Set

Instead of using only training and test sets, create:

```text
Training
Validation
Test
```

Use the validation set to select the number of PCA components.

Evaluate the final model only once using the test set.

### Exercise E — Unknown Person

Explain why PCA may still reconstruct the face of an unregistered person even though the recognition system cannot correctly identify that person.

---

## 17. Important Limitations

Eigenfaces is useful for understanding the fundamentals of face recognition, but it has several limitations.

Its performance can be strongly affected by:

- illumination
- face alignment
- pose
- facial expression
- camera differences
- background
- image resolution

PCA finds directions with high variance. These directions are **not necessarily the directions that best separate different identities**.

Modern face recognition systems therefore usually use deep neural networks and learned face embeddings. These methods will be introduced in later stages of the project.

---

## 18. Week 1 Key Concept

The most important pipeline to remember is:

```text
Image
  ↓
Face Detection
  ↓
Preprocessing
  ↓
Mean Subtraction
  ↓
PCA / Eigenfaces
  ↓
Feature Vector
  ↓
Feature Comparison
  ↓
Identity Prediction
  ↓
Evaluation
```

And remember:

> **Face reconstruction and face recognition are different tasks.**

---

## Reference

The PCA and Eigenfaces material is based in part on:

Steven L. Brunton and J. Nathan Kutz,  
**Data-Driven Science and Engineering**

Relevant sections:

```text
Sections 1.5–1.6
Pages 27–40
```

The notebook extends the textbook example with:

- train/test splitting
- nearest-neighbor face recognition
- recognition accuracy
- confusion matrices
- unknown-person rejection
- webcam face enrollment and recognition

---

## Educational Use

This repository is intended for educational purposes.

Face images and biometric data should be handled carefully. Students should only capture and use face images with appropriate permission or consent.

The Week 1 webcam system is an introductory teaching demonstration and is **not intended for security, authentication, or real-world deployment**.
