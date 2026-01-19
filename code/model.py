import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LayerNormalization
from tensorflow.keras.models import Model


def create_multi_label_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # 改进1：残差注意力模块
    def residual_attention(x):
        attn = tf.keras.layers.Attention(use_scale=True)([x, x])
        # 残差连接 + 门控机制
        gate = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        return x * gate + attn * (1 - gate)

    # 改进2：动态特征缩放
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = residual_attention(x)

    # 改进3：自适应特征压缩
    x = Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.AlphaDropout(0.38)(x)

    # 改进4：渐进式降维
    x = Dense(64, activation='selu', kernel_initializer='lecun_normal')(x)
    x = LayerNormalization(epsilon=1e-6)(x)

    x = Dense(32, activation=tf.keras.activations.swish)(x)




    # 改进5：输出头
    outputs = Dense(num_classes, activation='sigmoid',
                    kernel_constraint=tf.keras.constraints.UnitNorm())(x)

    return Model(inputs=inputs, outputs=outputs)
