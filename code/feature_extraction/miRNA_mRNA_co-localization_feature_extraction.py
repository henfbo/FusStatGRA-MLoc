import csv
import numpy as np
import pandas as pd

def entropy(p):
    p = np.asarray(p, dtype=np.float32)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / (s + 1e-8)
    return float(-(p * np.log(p + 1e-8)).sum())

def gini(p):
    p = np.asarray(p, dtype=np.float32)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / (s + 1e-8)
    return float(1.0 - np.sum(p * p))   # 1 - sum(p^2)

if __name__ == '__main__':
    # ===== 你按自己路径改这里 =====
    miRNA_mRNA_path = '../../dataset/miRNA_mRNA_matrix.txt'      # (1041,2836) 0/1或带权
    mRNA_loc_path   = '../../dataset/mRNA_localization.txt'     # (2836,4) one-hot

    # 读取矩阵
    df_miRNA_mRNA = pd.read_csv(miRNA_mRNA_path, header=None, delimiter='\t')
    miRNA_mRNA_matrix = df_miRNA_mRNA.values

    df_mRNA_loc = pd.read_csv(mRNA_loc_path, header=None, delimiter='\t')
    mRNA_loc = df_mRNA_loc.values

    n_miRNA = miRNA_mRNA_matrix.shape[0]
    n_mRNA = miRNA_mRNA_matrix.shape[1]
    assert mRNA_loc.shape[0] == n_mRNA and mRNA_loc.shape[1] == 4, "mRNA_loc 维度应为 (2836,4)"

    features = []

    for i in range(n_miRNA):
        w_sum = np.zeros(4, dtype=np.float32)

        # interaction 强度加权汇总
        row = miRNA_mRNA_matrix[i]
        for j in range(n_mRNA):
            w = row[j]
            if w != 0:
                # mRNA j 属于哪个区室就加到对应区室
                # 若 mRNA_loc 是 one-hot，这里等价于加到一个区室；若 multi-hot，会分摊到多个区室
                for c in range(4):
                    if mRNA_loc[j, c] == 1:
                        w_sum[c] += float(w)

        total = float(w_sum.sum())
        if total > 0:
            ratio = w_sum / (total + 1e-8)
        else:
            ratio = np.zeros(4, dtype=np.float32)

        # ===== 统计特征（6维）=====
        max_ratio = float(ratio.max())
        ent = entropy(ratio)
        gin = gini(ratio)
        num_active = float(np.sum(ratio > 0))

        # Top1-Top2 margin：定位是否“明确”
        p_sorted = np.sort(ratio)[::-1]
        margin = float(p_sorted[0] - p_sorted[1]) if p_sorted.shape[0] >= 2 else float(p_sorted[0])

        # 有效区室数（entropy 的指数形式）
        n_eff = float(np.exp(ent))  # ∈ [1,4]

        feat = np.concatenate([ratio, [max_ratio, ent, gin, num_active, margin, n_eff]], axis=0)  # 4+6=10
        features.append(feat.tolist())

    out_path = "../../feature/miRNA_mRNA_co-localization_feature_plus10.csv"
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(features)

    print("Saved:", out_path, "shape =", (len(features), len(features[0])))
