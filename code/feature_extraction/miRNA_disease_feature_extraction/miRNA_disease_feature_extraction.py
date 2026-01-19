import csv
import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec

if __name__ == '__main__':
    # 读取 miRNA 与疾病的相互作用矩阵
    df_miRNA_dis = pd.read_csv('../../../dataset/miRNA_disease.csv', header=None)
    miRNA_dis_matrix = df_miRNA_dis.values
    print(miRNA_dis_matrix.shape)

    # 创建一个空图
    G = nx.Graph()

    # 添加 miRNA 节点
    for i in range(1041):
        G.add_node(f"miRNA_{i}")

    # 添加疾病节点
    for i in range(640):
        G.add_node(f"disease_{i}")

    # 遍历矩阵，添加边并设置边的权重
    for miRNA_index in range(1041):
        for disease_index in range(640):
            interaction = miRNA_dis_matrix[miRNA_index, disease_index]
            if interaction != 0:
                # 将交互强度作为边的权重
                G.add_edge(f"miRNA_{miRNA_index}", f"disease_{disease_index}", weight=interaction)

    # 创建 Node2Vec 对象，指定使用边的权重
    node2vec = Node2Vec(G, dimensions=128, walk_length=150, num_walks=200, workers=1, weight_key='weight')

    # 训练模型
    model = node2vec.fit()

    # 提取 miRNA 特征向量
    miRNA_features = {}
    for miRNA_node in range(1041):
        miRNA_node_str = f"miRNA_{miRNA_node}"
        miRNA_features[miRNA_node_str] = model.wv[miRNA_node_str]

    # 将特征向量转换为 DataFrame
    df = pd.DataFrame.from_dict(miRNA_features, orient='index')
    df.columns = [f'Dimension_{i}' for i in range(128)]  # 列名可以根据需要自定义

    # 保存特征向量到 CSV 文件
    df.to_csv('../../../feature/miRNA_disease_feature_128_weight.csv')