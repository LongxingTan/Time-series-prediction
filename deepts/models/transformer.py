
import tensorflow as tf
from deepts.layers.attention_layer import *
# https://github.com/maxjcohen/transformer

params={
    'window_sizes':10,
    'predict_window_sizes':5,
    'n_layers':6,
    'attention_hidden_size':64*8,
    'num_heads':8,
    'ffn_hidden_size':64*8,
    'ffn_filter_size':64*8,
    'attention_dropout':0.1,
    'relu_dropout':0.1,
    'layer_postprocess_dropout':0.1,

}


class Transformer(object):
    def __init__(self):
        super(Transformer,self).__init__()
        self.params=params
        self.embedding_layer=EmbeddingLayer(embedding_size=params['attention_hidden_size'])
        self.position_encoding_layer=PositionEncoding(max_len=params['window_sizes'])
        self.position_encoding_layer_2=PositionEncoding(max_len=params['predict_window_sizes'])
        self.encoder_stack=EncoderStack(params)
        self.decoder_stack=DecoderStack(params)
        self.projection = tf.keras.layers.Dense(units=1)

    def get_config(self):
        return {}

    def __call__(self, inputs_shape,training=False):
        if training:
            x=tf.keras.layers.Input(inputs_shape)
            src_mask = self.get_src_mask(x)  # => batch_size * sequence_length
            src_mask = self.get_src_mask_bias(src_mask)  # => batch_size * 1 * 1 * input_sequence_length

            memory=self.encoder(encoder_inputs=x,mask=src_mask,training=training)

            targets=tf.keras.layers.Input([self.params['predict_window_sizes'],1])
            decoder_output = self.decoder(targets,memory,src_mask,training=training)
            outputs=self.projection(decoder_output)
            print(tf.math.is_nan(outputs))

            return tf.keras.Model([x,targets],outputs)
        else:
            x = tf.keras.layers.Input(inputs_shape)
            src_mask = self.get_src_mask(x)  # => batch_size * sequence_length
            src_mask = self.get_src_mask_bias(src_mask)  # => batch_size * 1 * 1 * input_sequence_length

            memory = self.encoder(encoder_inputs=x, mask=src_mask, training=training)

            decoder_inputs = tf.ones((x.shape[0], 1, 1), tf.int32)

            for _ in range(self.params['predict_window_sizes']):
                decoder_inputs_update=self.decoder(decoder_inputs,memory,src_mask,training)
                decoder_inputs=tf.concat([decoder_inputs,decoder_inputs_update],axis=1)


    def encoder(self,encoder_inputs, mask,training):
        '''

        :param inputs: sequence_inputs, batch_size * sequence_length * feature_dim
        :param training:
        :return:
        '''
        with tf.name_scope("encoder"):
            src=self.embedding_layer(encoder_inputs)  # batch_size * sequence_length * embedding_size
            src+=self.position_encoding_layer(src)

            if training:
                src=tf.nn.dropout(src,rate=0.01)  # batch_size * sequence_length * attention_hidden_size

            return self.encoder_stack(src,mask,training)

    def decoder(self,targets,memory,src_mask,training):
        with tf.name_scope("shift_targets"):
            # Shift targets to the right, and remove the last element
            decoder_inputs = tf.pad(targets, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
            tgt_mask=self.get_tgt_mask_bias(self.params['predict_window_sizes'])
            tgt=self.embedding_layer(decoder_inputs)

        with tf.name_scope("add_pos_encoding"):
            pos_encoding = self.position_encoding_layer_2(tgt)
            tgt += pos_encoding

        if training:
            tgt = tf.nn.dropout(tgt, rate=self.params["layer_postprocess_dropout"])
        with tf.name_scope('decoder'):
            logits=self.decoder_stack(tgt,memory,src_mask,tgt_mask,training)  #Todo：mask
        return logits

    def predict(self):
        pass

    def get_src_mask(self,x,pad=0):
        src_mask = tf.reduce_all(tf.math.equal(x, pad),axis=-1)
        return src_mask

    def get_src_mask_bias(self,mask):
        attention_bias = tf.cast(mask, tf.float32)
        attention_bias = attention_bias * tf.constant(-1e9, dtype=tf.float32)
        attention_bias = tf.expand_dims(tf.expand_dims(attention_bias, 1),1)  # => batch_size * 1 * 1 * input_length
        return attention_bias

    def get_tgt_mask_bias(self,length):
        valid_locs = tf.linalg.band_part(tf.ones([length, length], dtype=tf.float32),-1, 0)
        valid_locs = tf.reshape(valid_locs, [1, 1, length, length])
        decoder_bias = -1e9 * (1.0 - valid_locs)
        return decoder_bias


class EncoderStack(tf.keras.layers.Layer):
    def __init__(self,params):
        super(EncoderStack, self).__init__()
        self.params=params
        self.layers=[]

    def build(self,input_shape):
        for _ in range(self.params['n_layers']):
            attention_layer=Attention(self.params['attention_hidden_size'],self.params['num_heads'],self.params['attention_dropout'])
            feed_forward_layer=FeedForwardNetwork(self.params['ffn_hidden_size'],self.params['ffn_filter_size'],self.params['relu_dropout'])
            post_attention_layer=SublayerConnection(attention_layer,self.params)
            post_feed_forward_layer=SublayerConnection(feed_forward_layer,self.params)
            self.layers.append([post_attention_layer,post_feed_forward_layer])
        self.output_norm=tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype="float32")
        super(EncoderStack,self).build(input_shape)

    def get_config(self):
        return {
        }

    def call(self,encoder_inputs, src_mask, training):
        for n, layer in enumerate(self.layers):
            attention_layer = layer[0]
            ffn_layer = layer[1]

            with tf.name_scope('layer_{}'.format(n)):
                with tf.name_scope('self_attention'):
                    encoder_inputs = attention_layer(encoder_inputs,encoder_inputs, src_mask, training=training)
                with tf.name_scope('ffn'):
                    encoder_inputs = ffn_layer(encoder_inputs, training=training)

        return self.output_norm(encoder_inputs)


class DecoderStack(tf.keras.layers.Layer):
    def __init__(self,params):
        super(DecoderStack,self).__init__()
        self.params=params
        self.layers=[]

    def build(self,input_shape):
        for _ in range(self.params['n_layers']):
            self_attention_layer=Attention(self.params['attention_hidden_size'],self.params['num_heads'],self.params['attention_dropout'])
            enc_dec_attention_layer=Attention(self.params['attention_hidden_size'],self.params['num_heads'],self.params['attention_dropout'])
            feed_forward_layer=FeedForwardNetwork(self.params['ffn_hidden_size'],self.params['ffn_filter_size'],self.params['relu_dropout'])
            post_self_attention_layer=SublayerConnection(self_attention_layer,self.params)
            post_enc_dec_attention_layer=SublayerConnection(enc_dec_attention_layer,self.params)
            post_feed_forward_layer=SublayerConnection(feed_forward_layer,self.params)
            self.layers.append([post_self_attention_layer,post_enc_dec_attention_layer,post_feed_forward_layer])
        self.output_norm=tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype="float32")
        super(DecoderStack,self).build(input_shape)

    def get_config(self):
        return {
            'params':self.params
        }

    def call(self, decoder_inputs,encoder_outputs,src_mask,tgt_mask,training,cache=None,):
        for n, layer in enumerate(self.layers):
            self_attention_layer = layer[0]
            enc_dec_attention_layer = layer[1]
            ffn_layer = layer[2]

            #layer_cache = cache[layer_name] if cache is not None else None
            with tf.name_scope("dec_layer_{}".format(n)):
                with tf.name_scope('self_attention'):
                    decoder_inputs = self_attention_layer(decoder_inputs,decoder_inputs,tgt_mask,training=training)
                with tf.name_scope('enc_dec_attention'):
                    decoder_inputs = enc_dec_attention_layer(decoder_inputs,encoder_outputs,src_mask,training=training)  # Todo: mask??
                with tf.name_scope('ffn'):
                    decoder_inputs = ffn_layer(decoder_inputs,training=training)
        return self.output_norm(decoder_inputs)
