import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec

def entropy_of_dist(v):
    v = np.asarray(v, dtype=np.float32)
    s = v.sum()
    if s <= 0:
        return 0.0
    p = v / (s + 1e-8)
    return float(-(p * np.log(p + 1e-8)).sum())

def seq_similarity_stats(S, topk=20, thr=0.8):
    """
    S: (N,N) 序列相似度矩阵
    返回 (N,8) 统计特征
    """
    N = S.shape[0]
    feats = np.zeros((N, 8), dtype=np.float32)

    for i in range(N):
        row = S[i].copy()
        row[i] = 0.0

        # 基本统计
        row_sum = float(np.sum(row))
        row_max = float(np.max(row))
        nnz = float(np.sum(row > 0))
        high = float(np.sum(row >= thr))

        # top-k 统计
        if topk > 0:
            k = min(topk, N-1)
            top = np.partition(row, -k)[-k:]  # 无序 topk
            top_mean = float(np.mean(top))
            top_std = float(np.std(top))
            top_ent = entropy_of_dist(np.maximum(top, 0.0))
        else:
            top_mean = top_std = top_ent = 0.0

        # 集中度：1 - sum(p^2)，p 为归一化相似度
        if row_sum > 0:
            p = row / (row_sum + 1e-8)
            gini = float(1.0 - np.sum(p * p))
        else:
            gini = 0.0

        feats[i] = np.array([row_sum, row_max, nnz, high, top_mean, top_std, top_ent, gini], dtype=np.float32)

    return feats

def build_node2vec_seq_emb(sim_mat, out_dim=64):
    G = nx.Graph()
    N = sim_mat.shape[0]
    for i in range(N):
        G.add_node(str(i))

    # 建边：只保留 >0 的相似度
    for i in range(N):
        for j in range(i + 1, N):
            w = sim_mat[i, j]
            if w > 0:
                G.add_edge(str(i), str(j), weight=float(w))

    node2vec = Node2Vec(
        G, dimensions=out_dim, walk_length=150, num_walks=200,
        workers=1, weight_key='weight'
    )
    model = node2vec.fit()
    emb = np.vstack([model.wv[str(i)] for i in range(N)])  # (N,out_dim)
    return emb

if __name__ == '__main__':
    # ===== 你按自己路径改这里 =====
    sim_path = '../../dataset/miRNA_seq_sim.csv'

    S = pd.read_csv(sim_path, header=None).values.astype(np.float32)  # (1041,1041)

    emb64 = build_node2vec_seq_emb(S, out_dim=64)            # (1041,64)
    stats8 = seq_similarity_stats(S, topk=20, thr=0.8)       # (1041,8)

    feat72 = np.concatenate([emb64, stats8], axis=1)         # (1041,72)

    out_path = '../../feature/miRNA_seq_feature_64_plus_stats8.csv'
    pd.DataFrame(feat72).to_csv(out_path, index_label='Index')
    print("Saved:", out_path, "shape =", feat72.shape)
