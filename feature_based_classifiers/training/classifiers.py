from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import pysam
from extract_features import load_modkit_positions, get_methylation_features, build_dataset
import joblib
import os

methylation_cutoff = 0.8
compute_cpg = True
compute_mean = False
min_len_read = 500

# LOAD training BAM FILES
human_bam = "../human_ecoli_classification_dataset/training/human/bam/human.bam"
ecoli_bam = "../human_ecoli_classification_dataset/training/ecoli/bam/ecoli.bam"

# LOAD training bed files from modkit (already filtered by methylation cutoff > 0.5)
human_5mC = "../human_ecoli_classification_dataset/training/human/bed/human_5mC.bed"
human_6mA = "../human_ecoli_classification_dataset/training/human/bed/human_6mA.bed"
human_4mC = "../human_ecoli_classification_dataset/training/human/bed/human_4mC.bed"
ecoli_5mC = "../human_ecoli_classification_dataset/training/ecoli/bed/ecoli_5mC.bed"
ecoli_6mA = "../human_ecoli_classification_dataset/training/ecoli/bed/ecoli_6mA.bed"
ecoli_4mC = "../human_ecoli_classification_dataset/training/ecoli/bed/ecoli_4mC.bed"


X_human, y_human = build_dataset(human_bam, human_5mC, human_6mA, human_4mC, label=0, methylation_cutoff=methylation_cutoff, min_len_read=min_len_read, compute_cpg=compute_cpg, compute_mean=compute_mean)
X_ecoli, y_ecoli = build_dataset(ecoli_bam, ecoli_5mC, ecoli_6mA, ecoli_4mC, label=1, methylation_cutoff=methylation_cutoff, min_len_read=min_len_read, compute_cpg=compute_cpg, compute_mean=compute_mean)


# Downsample E. coli reads to balance classes
n_human = X_human.shape[0]
X_ecoli_shuf, y_ecoli_shuf = shuffle(
    X_ecoli, y_ecoli, random_state=42
)
X_ecoli_bal = X_ecoli_shuf[:n_human]
y_ecoli_bal = y_ecoli_shuf[:n_human]


X = np.vstack([X_human, X_ecoli_bal])
y = np.concatenate([y_human, y_ecoli_bal])

print ("Feature matrix shape:", X.shape)
print ("Labels shape:", y.shape)    
print("number of human samples (> {}bp long): {}".format(min_len_read, n_human))
print("number of ecoli samples (> {}bp long): {}".format(min_len_read, X_ecoli_bal.shape[0]))

X, y = shuffle(X, y, random_state=42)

classifiers = {
    "Logistic_Regression": LogisticRegression(max_iter=1000),
    "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale")),
    "Random_Forest": RandomForestClassifier(n_estimators=200, random_state=42)
}

# save the models in a model directory
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

# Train and save
for name, clf in classifiers.items():
    clf.fit(X, y)
    model_path = os.path.join(
        models_dir,
        f"{name}_model_cpg_{compute_cpg}_mean_{compute_mean}_cutoff_{methylation_cutoff}.joblib"
    )
    joblib.dump(clf, model_path)
    print(f"Saved {name} model to {model_path}")