import pysam
import numpy as np
import csv
import os
import random

def load_modkit_positions(path, prob_threshold, normalize):
    """
    Load modkit BED or CSV for one modification.
    Returns dict: { read_id : { pos : probability } }
    """
    meth = {}
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            read_id = row["read_id"]
            pos = int(row["forward_read_position"])

            prob = None
            if "call_prob" in row and row["call_prob"]:
                if normalize:
                    prob = 1.0
                else:             
                    prob = float(row["call_prob"])
            elif "mod_qual" in row and row["mod_qual"]:
                if normalize:
                    prob = 1.0
                else:
                    prob = float(row["mod_qual"])
            if prob is None or prob < prob_threshold:
                continue
            if read_id not in meth:
                meth[read_id] = {}
            meth[read_id][pos] = prob

    return meth


def get_multi_modification_vectors(read, meth_5mC, meth_6mA, meth_4mC, normalize, test):
    read_len = read.query_length
    vec_5mC = np.zeros(read_len, dtype=np.float32)
    vec_6mA = np.zeros(read_len, dtype=np.float32)
    vec_4mC = np.zeros(read_len, dtype=np.float32)

    read_id = read.query_name
    for mod_dict, vec in zip([meth_5mC, meth_6mA, meth_4mC],
                             [vec_5mC, vec_6mA, vec_4mC]):
        if read_id in mod_dict:
            for pos, prob in mod_dict[read_id].items():
                if 0 <= pos < read_len:
                    vec[pos] = prob

    return vec_5mC, vec_6mA, vec_4mC


def one_hot_encode(seq):
    """
    making four channels for DNA 
    """
    mapping = {"A":0, "C":1, "G":2, "T":3}
    arr = np.zeros((4, len(seq)), dtype=np.float32)
    for i, base in enumerate(seq):
        if base in mapping:
            arr[mapping[base], i] = 1.0
    return arr


def build_masked_tensor_multi(seq, vec_5mC, vec_6mA, vec_4mC, window_size, test, supreme_test):
    onehot = one_hot_encode(seq)
    masked_onehot = np.zeros_like(onehot)

    read_len = len(seq)
    combined_meth = np.maximum.reduce([vec_5mC, vec_6mA, vec_4mC])

    if test:
        masked_onehot = np.zeros_like(onehot)

        n_sites = np.count_nonzero(combined_meth)
        if supreme_test:
            n_sites = 10
        if n_sites > 0:
            random_sites = np.random.randint(0, read_len, size=n_sites)
            for i in random_sites:
                start = max(0, i - window_size)
                end = min(read_len, i + window_size + 1)
                masked_onehot[:, start:end] = onehot[:, start:end]
        vec_5mC = np.zeros_like(vec_5mC)
        vec_6mA = np.zeros_like(vec_6mA)
        vec_4mC = np.zeros_like(vec_4mC)

        mod_channels = np.vstack([vec_5mC, vec_6mA, vec_4mC])
        return np.vstack([masked_onehot, mod_channels])

    for i, val in enumerate(combined_meth):
        if val > 0:
            start = max(0, i - window_size)
            end = min(read_len, i + window_size + 1)
            masked_onehot[:, start:end] = onehot[:, start:end]

    mod_channels = np.vstack([vec_5mC, vec_6mA, vec_4mC])
    return np.vstack([masked_onehot, mod_channels])  # (7, read_len)


def fix_read_length(tensor, fixed_len=5000):
    """
    tensor: shape (7, L)
    returns: shape (7, fixed_len)
    """
    channels, L = tensor.shape
    if L < fixed_len:
        pad = np.zeros((channels, fixed_len - L), dtype=np.float32)
        return np.hstack([tensor, pad])
    elif L > fixed_len:
        start = np.random.randint(0, L - fixed_len + 1)
        return tensor[:, start:start + fixed_len]
    else:
        return tensor


def process_sample(bam_path, mod_5mC_path, mod_6mA_path, mod_4mC_path, methylation_cutoff, label_name, window_size, min_len, normalize, test, supreme_test, output_dir):
    """
    Creates fixed-length tensors for all reads and saves them to output_dir.
    """
    print(f"Processing sample: {label_name}")
    meth_5mC = load_modkit_positions(mod_5mC_path, methylation_cutoff, normalize)
    meth_6mA = load_modkit_positions(mod_6mA_path, methylation_cutoff, normalize)
    meth_4mC = load_modkit_positions(mod_4mC_path, methylation_cutoff, normalize)
    bamfile = pysam.AlignmentFile(bam_path, "rb", check_sq=False)
    count = 0

    for read in bamfile:
        seq = read.query_sequence
        if seq is None or len(seq) < min_len:
            continue

        vec_5mC, vec_6mA, vec_4mC = get_multi_modification_vectors(read, meth_5mC, meth_6mA, meth_4mC, normalize, test)
        tensor = build_masked_tensor_multi(seq, vec_5mC, vec_6mA, vec_4mC, window_size, test, supreme_test)
        tensor = fix_read_length(tensor)

        sample_dir = os.path.join(output_dir, label_name)
        os.makedirs(sample_dir, exist_ok=True)
        out_path = os.path.join(sample_dir, f"{label_name}_{read.query_name}.npy")
        np.save(out_path, tensor)
        count += 1

        if count % 1000 == 0:
            print(f"  saved {count} reads")

    bamfile.close()
    print(f"Done ({count} reads processed).")


def run_generation_training(
    window_size,
    methylation_cutoff,
    random_sequences=False,
    supreme_test=False,
    normalize=False,
    ):
    min_len = 500

    output_dir = (
        f"training_data_window{window_size}_cutoff{methylation_cutoff}"
        + ("_norm" if normalize else "")
        + ("_supreme" if supreme_test else "")
        + ("_test" if random_sequences else "")
        + "/"
    )
    os.makedirs(output_dir, exist_ok=True)

    process_sample(
        bam_path="../human_ecoli_classification_dataset/training/ecoli/bam/ecoli.bam",
        mod_5mC_path="../human_ecoli_classification_dataset/training/ecoli/bed/ecoli_5mC.bed",
        mod_6mA_path="../human_ecoli_classification_dataset/training/ecoli/bed/ecoli_6mA.bed",
        mod_4mC_path="../human_ecoli_classification_dataset/training/ecoli/bed/ecoli_4mC.bed",
        methylation_cutoff=methylation_cutoff,
        label_name="ecoli",
        window_size=window_size,
        min_len=min_len, 
        normalize=normalize,
        test=random_sequences,
        supreme_test=supreme_test,
        output_dir=output_dir
    )

    process_sample(
        bam_path="../human_ecoli_classification_dataset/training/human/bam/human.bam",
        mod_5mC_path="../human_ecoli_classification_dataset/training/human/bed/human_5mC.bed",
        mod_6mA_path="../human_ecoli_classification_dataset/training/human/bed/human_6mA.bed",
        mod_4mC_path="../human_ecoli_classification_dataset/training/human/bed/human_4mC.bed",
        methylation_cutoff=methylation_cutoff,
        label_name="human",
        window_size=window_size,
        min_len=min_len,
        normalize=normalize,    
        test=random_sequences,
        supreme_test=supreme_test,
        output_dir=output_dir
    )

def run_generation_test(
    window_size,
    methylation_cutoff,
    random_sequences=False,
    supreme_test=False,
    normalize=False,
    ):
    min_len = 500

    output_dir = (
        f"test_data_window{window_size}_cutoff{methylation_cutoff}"
        + ("_norm" if normalize else "")
        + ("_supreme" if supreme_test else "")
        + ("_test" if random_sequences else "")
        + "/"
    )
    os.makedirs(output_dir, exist_ok=True)

    process_sample(
        bam_path="../human_ecoli_classification_dataset/test/ecoli/bam/ecoli.bam",
        mod_5mC_path="../human_ecoli_classification_dataset/test/ecoli/bed/ecoli_5mC.bed",
        mod_6mA_path="../human_ecoli_classification_dataset/test/ecoli/bed/ecoli_6mA.bed",
        mod_4mC_path="../human_ecoli_classification_dataset/test/ecoli/bed/ecoli_4mC.bed",
        methylation_cutoff=methylation_cutoff,
        label_name="ecoli",
        window_size=window_size,
        min_len=min_len, 
        normalize=normalize,
        test=random_sequences,
        supreme_test=supreme_test,
        output_dir=output_dir
    )

    process_sample(
        bam_path="../human_ecoli_classification_dataset/test/human/bam/human.bam",
        mod_5mC_path="../human_ecoli_classification_dataset/test/human/bed/human_5mC.bed",
        mod_6mA_path="../human_ecoli_classification_dataset/test/human/bed/human_6mA.bed",
        mod_4mC_path="../human_ecoli_classification_dataset/test/human/bed/human_4mC.bed",
        methylation_cutoff=methylation_cutoff,
        label_name="human",
        window_size=window_size,
        min_len=min_len,
        normalize=normalize,    
        test=random_sequences,
        supreme_test=supreme_test,
        output_dir=output_dir
    )
