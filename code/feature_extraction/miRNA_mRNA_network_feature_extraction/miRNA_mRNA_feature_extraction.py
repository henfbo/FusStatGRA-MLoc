import pandas as pd
import networkx as nx
from node2vec import Node2Vec

if __name__ == '__main__':
    # 读取miRNA与mRNA的相互作用矩阵
    df_miRNA_mRNA = pd.read_csv('../../../dataset/miRNA_mRNA_matrix.txt', header=None, delimiter='\t')
    miRNA_mRNA_matrix = df_miRNA_mRNA.values
    print(miRNA_mRNA_matrix.shape)

    # 创建一个空图
    G = nx.Graph()

    # 添加miRNA节点
    for i in range(1041):
        G.add_node(f"miRNA_{i}")

    # 添加mRNA节点
    for i in range(2836):
        G.add_node(f"mRNA_{i}")

    # 遍历相互作用矩阵，添加边并设置边的权重
    for miRNA_index in range(1041):
        for mRNA_index in range(2836):
            interaction = miRNA_mRNA_matrix[miRNA_index, mRNA_index]
            if interaction != 0:  # 只有非零交互才建立边
                # 将交互强度作为边的权重
                G.add_edge(f"miRNA_{miRNA_index}", f"mRNA_{mRNA_index}", weight=interaction)

    # 创建Node2Vec对象，设置dimensions=128并使用边权重
    node2vec = Node2Vec(G, dimensions=128, walk_length=150, num_walks=200, workers=1, weight_key='weight')

    # 训练模型
    model = node2vec.fit()

    # 提取miRNA特征向量
    miRNA_features = {}
    for miRNA_node in range(1041):
        miRNA_node_str = f"miRNA_{miRNA_node}"
        miRNA_features[miRNA_node_str] = model.wv[miRNA_node_str]

    # 将miRNA特征向量转换为DataFrame
    df = pd.DataFrame.from_dict(miRNA_features, orient='index')

    # 重命名DataFrame的列名，将每一维作为一列
    df.columns = [f'Dimension_{i}' for i in range(128)]  # 列名可以根据需要自定义

    # 保存结果到CSV文件
    df.to_csv('../../../feature/miRNA_mRNA_network_feature_128_weight.csv')
