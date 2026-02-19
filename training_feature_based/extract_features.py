import numpy as np
import csv
import pysam
import sys

def load_modkit_positions(path, prob_threshold):
    """
    Returns a methylation dictionary from the modkit (refined) output
    {read_id: {pos: prob}}
    """
    meth = {}
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            read_id = row["read_id"]
            pos = int(row["forward_read_position"])
            prob = float(row.get("call_prob") or row.get("mod_qual") or 0)
            if prob < prob_threshold:
                continue
            meth.setdefault(read_id, {})[pos] = prob
    return meth

def get_methylation_features(read, meth_5mC, meth_6mA, meth_4mC, methylation_cutoff, compute_cpg, compute_mean):
    """
    Returns 12 features per read: fraction, mean, variance for:
    5mC CpG, 5mC non-CpG, 6mA, 4mC
    or 4 features per read (only the methylation frequency) if compute_mean is False.
    """
    seq = read.query_sequence.upper()
    read_len = len(seq)
    read_id = read.query_name
    features = []

    
    # 5mC 
    vec_5mC = np.zeros(read_len, dtype=np.float32)
    if read_id in meth_5mC:
        for pos, prob in meth_5mC[read_id].items():
            if 0 <= pos < read_len and seq[pos] == "C":
                vec_5mC[pos] = prob

    # CpG-specific
    if compute_cpg:
        cpg_positions = [i for i in range(len(seq)-1) if seq[i] == "C" and seq[i+1] == "G"]
        methylated_cpg = [pos for pos in cpg_positions if vec_5mC[pos] >= methylation_cutoff]
        fraction_cpg = len(methylated_cpg) / len(cpg_positions) if cpg_positions else 0.0

        if compute_mean:
            mean_cpg = np.mean([vec_5mC[pos] for pos in methylated_cpg]) if methylated_cpg else 0.0
            var_cpg = np.var([vec_5mC[pos] for pos in methylated_cpg]) if methylated_cpg else 0.0
            features.extend([fraction_cpg, mean_cpg, var_cpg])
        else:
            features.extend([fraction_cpg])

    methylated_vals = vec_5mC[vec_5mC >= methylation_cutoff]
    denom = seq.count("C")
    fraction = len(methylated_vals) / denom if denom > 0 else 0.0
    if compute_mean:            
        mean_prob = np.mean(methylated_vals) if len(methylated_vals) > 0 else 0.0
        var_prob = np.var(methylated_vals) if len(methylated_vals) > 0 else 0.0
        features.extend([fraction, mean_prob, var_prob])
    else:
        features.extend([fraction])
        
    # 6mA 
    vec_6mA = np.zeros(read_len, dtype=np.float32)
    if read_id in meth_6mA:
        for pos, prob in meth_6mA[read_id].items():
            if 0 <= pos < read_len and seq[pos] == "A":
                vec_6mA[pos] = prob
    methylated_vals = vec_6mA[vec_6mA >= methylation_cutoff]
    denom = seq.count("A")
    fraction = len(methylated_vals) / denom if denom > 0 else 0.0
    if compute_mean:            
         mean_prob = np.mean(methylated_vals) if len(methylated_vals) > 0 else 0.0
         var_prob = np.var(methylated_vals) if len(methylated_vals) > 0 else 0.0
         features.extend([fraction, mean_prob, var_prob])
    else:
        features.extend([fraction])

    # 4mC 
    vec_4mC = np.zeros(read_len, dtype=np.float32)
    if read_id in meth_4mC:
        for pos, prob in meth_4mC[read_id].items():
            if 0 <= pos < read_len and seq[pos] == "C":
                vec_4mC[pos] = prob
    methylated_vals = vec_4mC[vec_4mC >= methylation_cutoff]
    denom = seq.count("C")
    fraction = len(methylated_vals) / denom if denom > 0 else 0.0
    if compute_mean:
        mean_prob = np.mean(methylated_vals) if len(methylated_vals) > 0 else 0.0
        var_prob = np.var(methylated_vals) if len(methylated_vals) > 0 else 0.0
        features.extend([fraction, mean_prob, var_prob])
    else:               
        features.extend([fraction])
    return np.array(features, dtype=np.float32)  


def build_dataset(bam_path, mod_5mC_path, mod_6mA_path, mod_4mC_path, label, methylation_cutoff, min_len_read=0, compute_cpg = True, compute_mean = True):
    meth_5mC = load_modkit_positions(mod_5mC_path, methylation_cutoff)
    meth_6mA = load_modkit_positions(mod_6mA_path, methylation_cutoff)
    meth_4mC = load_modkit_positions(mod_4mC_path, methylation_cutoff)

    bamfile = pysam.AlignmentFile(bam_path, "rb", check_sq=False)
    X, y = [], []
    for read in bamfile:
        if len(read.query_sequence) <= min_len_read:
            continue
        features = get_methylation_features(read, meth_5mC, meth_6mA, meth_4mC, methylation_cutoff, compute_cpg, compute_mean)
        X.append(features)
        y.append(label)        
    bamfile.close()
    return np.array(X), np.array(y)
