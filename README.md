# Mobile-Money-Fraud-Detection

This project explores a **hybrid approach to fraud detection** by combining **supervised machine learning models** (Random Forest, XGBoost) with an **unsupervised Autoencoder (AE)** to detect anomalies.  
The goal is to improve detection of fraudulent transactions that traditional models might miss while maintaining high overall accuracy.

---

## 🧭 Project Overview

**Objective:**  
To build a hybrid fraud detection pipeline capable of identifying both known and unknown fraud patterns using structured transaction data.

**Core Idea:**  
1. **Supervised Models** (Random Forest, XGBoost) — learn from labeled data to classify known fraud patterns.  
2. **Unsupervised Model** (Autoencoder) — reconstructs normal transactions; high reconstruction error flags potential anomalies.  
3. **Hybrid Integration** — combines predictions and anomaly flags for improved overall detection accuracy.

---

## 📁 Repository Structure

```

fraud-detection-hybrid-ml/
│
├── data/
│   ├── README.md                  # Brief note on data sources and preprocessing
│   └── sample_data.csv            # code loads data directly from kaggle directory
│
├── notebooks/
│   ├── 1_EDA_and_Feature_Engineering.ipynb
│   ├── 2_Supervised_Learning_Models.ipynb
│   ├── 3_Unsupervised_Autoencoder.ipynb
│   └── 4_Hybrid_Model_Integration.ipynb
│
├── models/
│   ├── xgboost_model.pkl
│   ├── autoencoder_model.h5
│   └── random_forest_model.pkl       
│
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_supervised.py
│   ├── train_autoencoder.py
│   └── evaluate_hybrid.py
│
├── requirements.txt
├── README.md
└── LICENSE  (optional)

````

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/<Kwesisbits>/fraud-detection-hybrid-ml.git
cd fraud-detection-hybrid-ml
````

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate       # for macOS/Linux
venv\Scripts\activate          # for Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```
numpy
pandas
scikit-learn
xgboost
tensorflow
matplotlib
seaborn
joblib
```

### 4. Launch Jupyter/Colab

You can open and run the notebooks in:

* Google Colab (recommended)
* JupyterLab / VS Code notebooks

---

## 📓 Notebook Overview

| Notebook                                | Purpose                                                                                                                                 | Key Outputs                                                       |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| **1_EDA_and_Feature_Engineering.ipynb** | Explore dataset, visualize fraud distribution, and engineer new features (e.g., transaction velocity, amount ratios).                   | Cleaned & scaled datasets (`X_train_full_scaled`, `y_train_full`) |
| **2_Supervised_Learning_Models.ipynb**  | Train and evaluate Random Forest & XGBoost classifiers using class imbalance techniques (SMOTE, weighted loss).                         | `xgboost_model.pkl`                                               |
| **3_Unsupervised_Autoencoder.ipynb**    | Build and train an Autoencoder to detect anomalous transactions via reconstruction error.                                               | `autoencoder_model.h5`                                            |
| **4_Hybrid_Model_Integration.ipynb**    | Combine the AE anomaly scores with supervised predictions for a hybrid detection pipeline. Evaluate ROC-AUC, precision, recall, and F1. | `hybrid_results.json`                                             |

---

## 💾 Model Saving Commands

Each notebook saves its outputs automatically.
You can add or verify the following lines near the end of your notebooks:

```python
import os, joblib, json
os.makedirs("models", exist_ok=True)

# Save Supervised Model
joblib.dump(xgb, "models/xgboost_model.pkl")

# Save Autoencoder
autoencoder.save("models/autoencoder_model.h5")


```

---

## 📊 Evaluation Metrics

The hybrid system is evaluated using:

* **ROC-AUC** — overall model discrimination
* **Precision & Recall** — focus on minimizing false positives/negatives
* **F1 Score** — harmonic mean of precision and recall
* **Confusion Matrix** — visual comparison between predicted and actual fraud

---

## 🚀 Results Summary

| Model            | ROC-AUC   | Precision | Recall   | F1 Score                |
| ---------------- | --------- | --------- | -------- | ----------------------- |
| Random Forest    | ~0.93     | 0.89      | 0.29     | 0.43                    |
| XGBoost          | ~0.95     | 0.91      | 0.33     | 0.48                    |
| Autoencoder      | —         | —         | —        | Anomaly threshold-based |
| **Hybrid Model** | **~0.97** | **0.90**  | **0.36** | **0.51**                |

*(Values are approximate — may vary depending on data splits.)*

---

## 📈 Future Improvements

* Integrate explainability with **SHAP** or **LIME** for model interpretation.
* Deploy as an **API** using FastAPI or Streamlit.
* Implement **online learning** for continuous fraud pattern updates.

---

## 🧠 Key Learnings

* Combining unsupervised and supervised approaches improves rare-event detection.
* Autoencoders can successfully identify subtle fraud anomalies missed by traditional models.
* Feature scaling, imbalance handling, and threshold tuning are critical for fraud detection accuracy.

---

## 🧑‍💻 Author

**Nana Kwesi Amponsah**
AI Engineer | Data Science
📍 Ghana
📧 [nanaamponsah391@gmail.com](mailto:nanaamponsah391@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/nana-kwesi-amponsah/) | [GitHub](https://github.com/<Kwesisbits>)

---

## 🪪 License

This project is released under the [MIT License](LICENSE).

---

> “Fraud detection is not just about finding anomalies — it’s about learning what *normal* looks like.”

```

