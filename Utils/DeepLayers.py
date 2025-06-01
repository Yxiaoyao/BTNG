import tensorflow as tf


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_heads, d_model, dff, dropout_rate):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.attention_layers = [self._multi_head_attention(num_heads, d_model) for _ in range(num_layers)]
        self.dense_layers = [tf.layers.Dense(dff, activation='relu') for _ in range(num_layers)]
        self.output_layer = tf.layers.Dense(d_model)
        self.dropout = tf.layers.Dropout(dropout_rate)


    def _multi_head_attention(self, num_heads, d_model):
        # 自定义多头注意力机制
        def attention(query, key, value):
            attention_weights = tf.nn.softmax(tf.matmul(query, key, transpose_b=True) / tf.sqrt(float(d_model)), axis=-1)
            return tf.matmul(attention_weights, value)
        return attention

    def call(self, inputs, training=False):
        x = inputs
        for i in range(self.num_layers):
            # Multi-head Attention
            attention_output = self.attention_layers[i](x, x, x)
            attention_output = self.dropout(attention_output, training=training)
            x = x + attention_output

            # Feed-forward network
            dense_output = self.dense_layers[i](x)
            x = x + dense_output

        return self.output_layer(x)
