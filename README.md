# Contextual Text Labeling

## Overview

The goal of this project is to **label short text phrases based on contextual meaning**, for example:

- **“Blue Jeans Ltd.” → Company**
- **“blue jeans” → Physical Goods**

This distinction is challenging for traditional NLP libraries (e.g., spaCy), which often rely on surface-level patterns and struggle with short, ambiguous phrases.  

---

## Repository Structure

- **Train_dataset_building/**  - Scripts and resources for collecting and generating labeled training data.
- **Train_model.ipynb**  - Notebook used to train and evaluate the classification model.
- **Test_classifier/**  -  Inference pipeline for testing the trained model on user-provided input.

---

## Dataset Construction

The training dataset consists of the following classes:

| Label | Examples | Data Source |
|------|---------|------------|
| **Date** | 01/15/2019, 15/02/2018, Dec 24th 2016, April 9 2015 | Programmatically generated using datetime |
| **Location** | Beijing, USA, Boston, Hong Kong | Downloaded from public datasets |
| **Random String** | INV01354-017, HKU781234 | Generated using random patterns |
| **Company** | Private Inc Limited, Fisher Goods Ltd | Generated using custom functions |
| **Physical Goods** | toys, jeans, boxes, computer, biscuit | Collected from public product lists |
| **Other** | Any unmatched input | Not explicitly included; inferred when prediction confidence is low |

### Dataset Balancing

- Most classes were downsampled to 20,000 samples.
- The Physical Goods class contained approximately 5,500 raw samples.
- Instead of downsampling further, class weights were applied during training to address class imbalance.

---

## Model Training

- A **Logistic Regression** classifier was used as the baseline model.
  
- The model was implemented and stored as a **scikit-learn pipeline**, where:
  - Raw input strings are vectorized using **TF-IDF** (`TfidfVectorizer`).
  - The resulting feature vectors are passed directly to the **Logistic Regression** classifier.
  
- The complete pipeline (vectorizer + classifier) was serialized and saved as **`model.joblib`**, ensuring consistent preprocessing and inference.
- The resulting classification report showed extremely high performance.
<img width="415" height="210" alt="Screenshot 2026-02-04 at 19 47 21" src="https://github.com/user-attachments/assets/50f61c41-b7ff-4103-b8d0-ca25ea184599" />


### Model Diagnostics

To ensure reliability and rule out false positives, several validation checks were performed:

1. Data leakage test -  Initial 100% accuracy raised concerns; further checks confirmed no data leakage.
<img width="814" height="29" alt="Screenshot 2026-02-04 at 19 48 05" src="https://github.com/user-attachments/assets/a596266d-9b9f-4076-abc4-b3bd7ca3368c" />

2. Train–test overlap check  
<img width="230" height="52" alt="Screenshot 2026-02-04 at 19 48 26" src="https://github.com/user-attachments/assets/b73db445-2597-40a4-9101-714ee1a49779" />

3. Randomized label test  
<img width="388" height="44" alt="Screenshot 2026-02-04 at 19 48 57" src="https://github.com/user-attachments/assets/756066c2-2448-4fe1-9276-572844509b50" />

4. Class separation analysis -  Feature-space visualization showed strong separation between classes
 <img width="172" height="103" alt="Screenshot 2026-02-04 at 19 49 18" src="https://github.com/user-attachments/assets/5e404c72-b246-4b82-84cc-dbb68ffa7bbb" />



The strong performance is expected, as many classes exhibit clear linguistic patterns such as:  
Company names often include suffixes such as *Ltd*, *Inc*, or *Limited* OR Dates follow recognizable numeric or textual formats etc.

Given this clarity, further steps such as advanced model selection or hyperparameter optimization were intentionally skipped.

- Cross-validation confirmed the robustness and consistency of the model.
<img width="370" height="88" alt="Screenshot 2026-02-04 at 19 50 33" src="https://github.com/user-attachments/assets/0574b2d1-35e8-47be-8e14-29ad9d1ff09c" />

---

## Inference

The inference pipeline allows users to input a short text string and receive one of the following labels:

- Date  
- Location  
- Random String  
- Company  
- Physical Goods  
- Other  

### Running Inference

Instructions for running inference are provided in the **README** inside the `Text-classifier` folder.

---

