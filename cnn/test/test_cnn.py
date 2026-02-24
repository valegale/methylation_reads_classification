import torch
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from cnn_model import ThreeBranchCNN
import sys


def run_test(window_size, cutoff):
    normalize = False  
    test = False      
    supreme_test = True  
    device = "cuda" if torch.cuda.is_available() else "cpu"

    norm_str = "_norm" if normalize else ""

    model_path = f"training_data_window{window_size}_cutoff{cutoff}{norm_str}.pth"
    output_file = f"test_results_window{window_size}_cutoff{cutoff}{norm_str}.txt"

    human_dir = f"test_data_window{window_size}_cutoff{cutoff}{norm_str}/human/"
    ecoli_dir = f"test_data_window{window_size}_cutoff{cutoff}{norm_str}/ecoli/"

    model = ThreeBranchCNN(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()


    test_tensors = []
    test_labels = []

    # Human
    for fname in os.listdir(human_dir):
        if fname.endswith(".npy"):
            tensor = np.load(os.path.join(human_dir, fname))
            test_tensors.append(torch.tensor(tensor, dtype=torch.float32))
            test_labels.append(0)  # human = 0

    # E. coli
    for fname in os.listdir(ecoli_dir):
        if fname.endswith(".npy"):
            tensor = np.load(os.path.join(ecoli_dir, fname))
            test_tensors.append(torch.tensor(tensor, dtype=torch.float32))
            test_labels.append(1)  # ecoli = 1


    all_preds = []

    with torch.no_grad():
        for tensor in test_tensors:
            tensor = tensor.unsqueeze(0).to(device)  # add batch dimension (1, 7, L)
            out = model(tensor)
            pred = out.argmax(1).item()
            all_preds.append(pred)

    # Compute metrics
    accuracy = accuracy_score(test_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, all_preds, average="binary")
    conf_mat = confusion_matrix(test_labels, all_preds)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("Confusion matrix:")
    print(conf_mat)

    with open(output_file, "w") as f:
        f.write(f"Accuracy : {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall   : {recall:.4f}\n")
        f.write(f"F1-score : {f1:.4f}\n")
        f.write("Confusion matrix:\n")
        f.write(np.array2string(conf_mat))
        f.write("\n")

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    window_size = int(sys.argv[1])
    cutoff = float(sys.argv[2])
    run_test(window_size, cutoff)