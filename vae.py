import logging
import os

import numpy as np
import tensorflow as tf

class Encoder(tf.keras.layers.Layer):
    def __init__(self,
                 encoder_hidden_dims: list,
                 n_z: int,
                 **kwargs: dict):
        super().__init__(**kwargs)

        self.relu_fn = tf.nn.swish

        self.dense_layers = []
        for i, num in enumerate(encoder_hidden_dims):
            dense = tf.keras.layers.Dense(num, name='dense{}'.format(i), activation=self.relu_fn)
            self.dense_layers.append(dense)

        self.z_mu = tf.keras.layers.Dense(n_z)
        self.z_sigma = tf.keras.layers.Dense(n_z)


    def call(self, inputs: tf.Tensor, condition: tf.Tensor, training: bool):
        x = tf.concat([inputs, condition], -1)

        for dense in self.dense_layers:
            x = dense(x)

        mu = self.z_mu(x)
        sigma = self.z_sigma(x)

        return mu, sigma

class Decoder(tf.keras.layers.Layer):
    def __init__(self,
                 decoder_hidden_dims: list,
                 n_outputs: int,
                 **kwargs: dict):
        super().__init__(**kwargs)

        self.relu_fn = tf.nn.swish

        self.dense_layers = []
        for i, num in enumerate(decoder_hidden_dims):
            dense = tf.keras.layers.Dense(num, name='dense{}'.format(i), activation=self.relu_fn)
            self.dense_layers.append(dense)

        self.out = tf.keras.layers.Dense(n_outputs, activation='sigmoid')


    def call(self, inputs: tf.Tensor, training: bool):
        x = inputs
        for dense in self.dense_layers:
            x = dense(x)

        x = self.out(x)
        return x

class VAE(tf.keras.Model):
    def __init__(self,
                 encoder_hidden_dims: list,
                 num_z: int,
                 decoder_hidden_dims: list,
                 num_outputs: int,
                 **kwargs: dict):
        super().__init__(**kwargs)

        self.n_z = num_z

        self.rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1 / 255)

        self.encoder = Encoder(encoder_hidden_dims, num_z)
        self.decoder = Decoder(decoder_hidden_dims, num_outputs)


    def call(self, inputs: tf.Tensor, labels: tf.Tensor, training: bool):
        batch_size = tf.shape(inputs)[0]

        scaled_x = self.rescale(inputs)
        scaled_x = tf.reshape(scaled_x, [batch_size, -1])

        mu, sigma = self.encoder(scaled_x, labels, training)

        eps = tf.random.normal([batch_size, self.n_z], mean=0, stddev=1)
        z = mu + tf.math.exp(sigma / 2) * eps


        dec_input = tf.concat([z, labels], -1)
        dec_output = self.decoder(dec_input, training)

        return scaled_x, dec_output, mu, sigma

    @tf.function
    def generate(self, labels: tf.Tensor, output_dir: str):
        batch_size = tf.shape(labels)[0]

        if True:
            z = tf.zeros([batch_size, self.n_z], dtype=tf.float32)
        else:
            z = tf.random.normal([batch_size, self.n_z], mean=0, stddev=1)

        dec_input = tf.concat([z, labels], -1)
        dec_output = self.decoder(dec_input, training=False)
        n = tf.cast(dec_output.shape[1], tf.float32)
        n = tf.cast(tf.math.sqrt(n), tf.int32)
        dec_output = tf.reshape(dec_output, [batch_size, n, n, 1])
        grey_data = tf.cast(dec_output * 255, tf.uint8)

        indexes = tf.range(tf.shape(labels)[0])

        def save_image(index):
            filename = tf.strings.format('{}.png', index)
            filename = tf.strings.join([output_dir, '/', filename])

            data = grey_data[index, ...]
            img = tf.io.encode_png(data)
            tf.io.write_file(filename, img)
            return index

        tf.map_fn(save_image, indexes, parallel_iterations=16)

class Metric:
    def __init__(self):
        self.kl_loss = tf.keras.metrics.Mean(name='kl_loss')
        self.rec_loss = tf.keras.metrics.Mean(name='rec_loss')
        self.total_loss = tf.keras.metrics.Mean(name='total_loss')

    def reset_states(self):
        self.kl_loss.reset_states()
        self.rec_loss.reset_states()
        self.total_loss.reset_states()

    def str_result(self):
        return 'loss: kl: {:.4e}, rec: {:.4e}, total: {:.4e}'.format(self.kl_loss.result(), self.rec_loss.result(), self.total_loss.result())

class Loss:
    def __init__(self, from_logits: bool):
        self.train_metric = Metric()
        self.eval_metric = Metric()

        self.output_binary_ce = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits, reduction=tf.keras.losses.Reduction.SUM)

    def reset_states(self):
        self.train_metric.reset_states()
        self.eval_metric.reset_states()

    def z_kl_loss(self, mu, sigma):
        x = tf.math.exp(sigma) + tf.math.square(mu) - 1 - sigma
        x = 0.5 * tf.reduce_sum(x, -1)
        return x

    def __call__(self, y_true, y_pred, mu, sigma, training):
        rec_loss = self.output_binary_ce(y_true, y_pred)

        kl_loss = self.z_kl_loss(mu, sigma)

        m = self.train_metric if training else self.eval_metric
        m.kl_loss.update_state(kl_loss)
        m.rec_loss.update_state(rec_loss)

        total_loss = rec_loss + kl_loss
        m.total_loss.update_state(total_loss)

        return total_loss

def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = tf.cast(x_train, tf.float32)
    x_test = tf.cast(x_test, tf.float32)

    num_labels = 10
    y_train = tf.one_hot(y_train, num_labels)
    y_test = tf.one_hot(y_test, num_labels)

    batch_size = 512

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    encoder_hidden_dims = [512]
    num_latent_vars = 2

    decoder_hidden_dims = [512]
    decoder_output_dim = 28*28

    model = VAE(encoder_hidden_dims=encoder_hidden_dims, num_z=num_latent_vars, decoder_hidden_dims=decoder_hidden_dims, num_outputs=decoder_output_dim)
    model(tf.ones([batch_size, 28*28], dtype=tf.float32), tf.zeros([batch_size, num_labels], dtype=tf.float32), training=True)
    model.summary()

    generate_labels = tf.range(num_labels)
    generate_labels = tf.one_hot(generate_labels, num_labels)
    output_dir = tf.Variable('results_dense/start')
    if False:
        model.generate(generate_labels, output_dir)
        exit(0)

    opt = tf.keras.optimizers.Adam(lr=0.001)

    loss_object = Loss(from_logits=False)

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            scaled_x, y_pred, mu, sigma = model(images, labels, training=True)
            loss = loss_object(scaled_x, y_pred, mu, sigma, training=True)

        gradients = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(gradients, model.trainable_variables))

    @tf.function
    def test_step(images, labels):
        scaled_x, y_pred, mu, sigma = model(images, labels, training=False)
        loss = loss_object(scaled_x, y_pred, mu, sigma, training=False)

    num_epochs = 150
    for epoch in range(num_epochs):
        loss_object.reset_states()

        for images, labels in train_ds:
            train_step(images, labels)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        print('epoch {}, train: {}, eval: {}'.format(
                epoch,
                loss_object.train_metric.str_result(),
                loss_object.eval_metric.str_result(),
            ))

        output_dir.assign('results_dense/{}'.format(epoch))
        model.generate(generate_labels, output_dir)

if __name__ == '__main__':
    main()
