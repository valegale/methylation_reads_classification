import matplotlib.pyplot as plt
from extract_features import load_modkit_positions, get_methylation_features, build_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import joblib
import numpy as np
import csv
import pysam

# Model directory
model_dir = "models"

# LOAD test BAM FILES
human_bam = "../human_ecoli_classification_dataset/test/human/bam/human.bam"
ecoli_bam = "../human_ecoli_classification_dataset/test/ecoli/bam/ecoli.bam"

# LOAD test bed files from modkit (already filtered by methylation cutoff > 0.5)
human_5mC = "../human_ecoli_classification_dataset/test/human/bed/human_5mC.bed"
human_6mA = "../human_ecoli_classification_dataset/test/human/bed/human_6mA.bed"
human_4mC = "../human_ecoli_classification_dataset/test/human/bed/human_4mC.bed"
ecoli_5mC = "../human_ecoli_classification_dataset/test/ecoli/bed/ecoli_5mC.bed"
ecoli_6mA = "../human_ecoli_classification_dataset/test/ecoli/bed/ecoli_6mA.bed"
ecoli_4mC = "../human_ecoli_classification_dataset/test/ecoli/bed/ecoli_4mC.bed"


methylation_cutoff = 0.8
compute_cpg = True
compute_mean = False
min_len_read = 500

X_human, y_human = build_dataset(human_bam, human_5mC, human_6mA, human_4mC, label=0, methylation_cutoff=methylation_cutoff, min_len_read=min_len_read, compute_cpg=compute_cpg, compute_mean=compute_mean)
X_ecoli, y_ecoli = build_dataset(ecoli_bam, ecoli_5mC, ecoli_6mA, ecoli_4mC, label=1, methylation_cutoff=methylation_cutoff, min_len_read=min_len_read, compute_cpg=compute_cpg, compute_mean=compute_mean)

print ("Feature matrix shape:", X_human.shape, X_ecoli.shape)

model_files = {
    "Logistic Regression": f"Logistic_Regression_model_cpg_{compute_cpg}_mean_{compute_mean}_cutoff_{methylation_cutoff}.joblib",
    "SVM": f"SVM_model_cpg_{compute_cpg}_mean_{compute_mean}_cutoff_{methylation_cutoff}.joblib",
    "Random Forest": f"Random_Forest_model_cpg_{compute_cpg}_mean_{compute_mean}_cutoff_{methylation_cutoff}.joblib",
}

models = {}

for name, fname in model_files.items():
    model_path = os.path.join(model_dir, fname)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    models[name] = joblib.load(model_path)
    print(f"Loaded {name} from {model_path}")

# Build test set
X_test = np.vstack([X_human, X_ecoli])
y_test = np.concatenate([y_human, y_ecoli])

print("=== TEST SET ===")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

for name, clf in models.items():
    print(f"\n==============================")
    print(f"MODEL: {name}")
    print(f"==============================")

    # Predict
    y_pred = clf.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("Confusion matrix:")
    print(cm)

    # Error rates
    human_err = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    ecoli_err = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    print(f"Human → Ecoli error rate: {human_err:.4f}")
    print(f"Ecoli → Human error rate: {ecoli_err:.4f}")

    # Detailed stats
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

model_names = list(models.keys())
accuracies = []

for name, clf in models.items():
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

plt.figure(figsize=(6,4))
plt.bar(model_names, accuracies, color=['skyblue', 'salmon', 'lightgreen'])
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison on Test Set")
plt.ylim(0, 1)
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center', fontweight='bold')
plt.savefig("accuracy.png", dpi=300, bbox_inches='tight')