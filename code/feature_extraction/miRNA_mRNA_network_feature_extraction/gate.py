import tensorflow._api.v2.compat.v1 as tf


class GATE():

    def __init__(self, hidden_dims, lambda_,contrastive_loss_weight=0.1):
        self.lambda_ = lambda_
        self.contrastive_loss_weight = contrastive_loss_weight
        self.n_layers = len(hidden_dims) -1
        # 256 128
        self.W, self.v = self.define_weights(hidden_dims)
        self.C = {}

# X节点的特征表示，A图的邻接矩阵
    def __call__(self, A, X, R, S):

        # Encoder
        H = X
        for layer in range(self.n_layers):
            # 1041 128
            H = self.__encoder(A, H, layer)

        # Final node representations
        self.H = H

        # Decoder
        for layer in range(self.n_layers - 1, -1, -1):
            H = self.__decoder(H, layer)
        X_ = H

        # The reconstruction loss of node features
        features_loss = tf.sqrt(tf.reduce_sum(tf.reduce_sum(tf.pow(X - X_, 2))))

        # The reconstruction loss of the graph structure
        self.S_emb = tf.nn.embedding_lookup(self.H, S)
        self.R_emb = tf.nn.embedding_lookup(self.H, R)
        structure_loss = -tf.log(tf.sigmoid(tf.reduce_sum(self.S_emb * self.R_emb, axis=-1)))
        structure_loss = tf.reduce_sum(structure_loss)

        contrastive_loss = self.contrastive_loss(self.H, S, R)
        # Total loss
        self.loss = features_loss + self.lambda_ * structure_loss + self.contrastive_loss_weight * contrastive_loss

        return self.loss, self.H, self.C

    def contrastive_loss(self, embeddings, S, R):
        # Calculate cosine similarity between positive and negative pairs
        positive_pairs = tf.nn.embedding_lookup(embeddings, S)
        negative_pairs = tf.nn.embedding_lookup(embeddings, R)

        positive_sim = tf.reduce_sum(positive_pairs * negative_pairs, axis=1)
        contrastive_loss = tf.reduce_mean(tf.maximum(0.0, 1.0 - positive_sim))

        return contrastive_loss

    def __encoder(self, A, H, layer):
        H = tf.matmul(H, self.W[layer])
        self.C[layer] = self.graph_attention_layer(A, H, self.v[layer], layer)
        return tf.sparse_tensor_dense_matmul(self.C[layer], H)

    def __decoder(self, H, layer):
        H = tf.matmul(H, self.W[layer], transpose_b=True)
        return tf.sparse_tensor_dense_matmul(self.C[layer], H)

    def define_weights(self, hidden_dims):
        W = {}
        for i in range(self.n_layers):
            W[i] = tf.get_variable("W%s" % i, shape=(hidden_dims[i], hidden_dims[i+1]))

        Ws_att = {}
        for i in range(self.n_layers):
            v = {}
            v[0] = tf.get_variable("v%s_0" % i, shape=(hidden_dims[i+1], 1))
            v[1] = tf.get_variable("v%s_1" % i, shape=(hidden_dims[i+1], 1))
            Ws_att[i] = v

        return W, Ws_att

    def graph_attention_layer(self, A, M, v, layer):

        with tf.variable_scope("layer_%s"% layer):
            f1 = tf.matmul(M, v[0])
            f1 = A * f1
            f2 = tf.matmul(M, v[1])
            print(f2.shape, M.shape, v[1].shape)
            f2 = A * tf.transpose(f2, [1, 0])
            logits = tf.sparse_add(f1, f2)

            unnormalized_attentions = tf.SparseTensor(indices=logits.indices,
                                         values=tf.nn.sigmoid(logits.values),
                                         dense_shape=logits.dense_shape)
            attentions = tf.sparse_softmax(unnormalized_attentions)

            attentions = tf.SparseTensor(indices=attentions.indices,
                                         values=attentions.values,
                                         dense_shape=attentions.dense_shape)

            return attentions