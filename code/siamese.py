import tensorflow as tf

class siamese:

    # Create model
    def __init__(self, feature_size, embedding, word_id):
        self.x1 = tf.placeholder(tf.int32, [None, feature_size])
        self.x2 = tf.placeholder(tf.int32, [None, feature_size])
        self.pretrained_embeddings = embedding

        self.map_to_embedding()
        s1, s2 = self.sum_layer()

        with tf.variable_scope("siamese") as scope:
            self.o1 = self.network(s1)
            scope.reuse_variables()
            self.o2 = self.network(s2)

        # Create loss
        self.y = tf.placeholder(tf.float32, [None])
        self.loss = self.cosine_sim_loss()
        

    def map_to_embedding(self):
    	embeddingTensor = tf.Variable(self.pretrained_embeddings, tf.float32)
    	self.x1_embed = tf.nn.embedding_lookup(embeddingTensor, self.x1)
    	self.x2_embed = tf.nn.embedding_lookup(embeddingTensor, self.x2)

    def sum_layer(self):
    	s1 = tf.reduce_sum(self.x1_embed, axis = 1)
    	s2 = tf.reduce_sum(self.x2_embed, axis = 1)
    	return s1, s2

    def network(self, x):
        weights = []
        fc1 = self.fc_layer(x, 1024, "fc1")
        return fc1

    def fc_layer(self, bottom, n_weight, name):
        assert len(bottom.get_shape()) == 2
        n_prev_weight = bottom.get_shape()[1]
        initer = tf.truncated_normal_initializer(stddev=0.01)
        W = tf.get_variable(name+'W', dtype=tf.float64, shape=[n_prev_weight, n_weight], initializer=initer)
        b = tf.get_variable(name+'b', dtype=tf.float64, initializer=tf.constant(0.01, shape=[n_weight], dtype=tf.float64))
        fc = tf.nn.bias_add(tf.matmul(bottom, W), b)
        return fc

    def cosine_sim_loss(self):
        cosine_sim_nume = tf.reduce_sum(tf.multiply(self.o1, self.o2), axis = 1)
        cosine_sim_denomi = tf.norm(self.o1, axis = 1) * tf.norm(self.o2, axis = 1)
        self.cosine_sim = cosine_sim_nume / cosine_sim_denomi
        loss = tf.losses.mean_squared_error(self.y, self.cosine_sim)
        return loss