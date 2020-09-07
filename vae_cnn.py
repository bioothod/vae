import logging
import os

import numpy as np
import tensorflow as tf

import resnet

logger = logging.getLogger('gclass')
logger.propagate = False
logger.setLevel(logging.INFO)
__fmt = logging.Formatter(fmt='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%d/%m/%y %H:%M:%S')
__handler = logging.StreamHandler()
__handler.setFormatter(__fmt)
logger.addHandler(__handler)

class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, channels_out, **kwargs):
        super().__init__(**kwargs)

        self.relu_fn = tf.nn.swish

        channels_in = channels_out // 4

        self.preact_bn = tf.keras.layers.BatchNormalization()

        self.bn0 = tf.keras.layers.BatchNormalization()
        self.conv0 = tf.keras.layers.Conv2D(channels_in, kernel_size=1, strides=1, name='conv0', padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2D(channels_in, kernel_size=3, strides=2, name='conv1', padding='same')
        self.conv2 = tf.keras.layers.Conv2D(channels_out, kernel_size=1, strides=1, name='conv2', padding='same')

        self.shortcut = tf.keras.layers.Conv2D(channels_out, kernel_size=1, strides=2, name='shortcut', padding='same')

    def __call__(self, inputs, training=True):
        preact = self.preact_bn(inputs, training=training)
        preact = self.relu_fn(preact)

        shortcut = self.shortcut(preact)

        x = self.conv0(preact)
        x = self.bn0(x, training=training)
        x = self.relu_fn(x)

        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.relu_fn(x)

        x = self.conv2(x)

        x += shortcut

        return x

class EncoderBlock1(tf.keras.layers.Layer):
    def __init__(self, num_features, **kwargs):
        super().__init__(**kwargs)

        self.relu_fn = tf.nn.swish

        self.dense = tf.keras.layers.Dense(num_features, activation=self.relu_fn)

    def __call__(self, inputs, training=True):
        x = self.dense(inputs)
        return x

class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, channels_out, **kwargs):
        super().__init__(**kwargs)

        self.relu_fn = tf.nn.swish
        self.conv0 = tf.keras.layers.Conv2DTranspose(channels_out, kernel_size=3, strides=2, name='conv0', padding='same', activation=self.relu_fn)
        self.conv1 = tf.keras.layers.Conv2DTranspose(channels_out, kernel_size=3, strides=1, name='conv1', padding='same', activation=self.relu_fn)

    def __call__(self, inputs, training=True):
        x = inputs
        x = self.conv0(x)
        x = self.conv1(x)
        return x

class DecoderBlock1(tf.keras.layers.Layer):
    def __init__(self, num_features, **kwargs):
        super().__init__(**kwargs)

        self.relu_fn = tf.nn.swish

        self.dense = tf.keras.layers.Dense(num_features, activation=self.relu_fn)

    def __call__(self, inputs, training=True):
        batch_size = inputs.shape[0]
        x = tf.reshape(inputs, [batch_size, -1])
        x = self.dense(x)
        return x

class Encoder(tf.keras.layers.Layer):
    def __init__(self,
                 encoder_hidden_dims: list,
                 have_condition: bool,
                 input_shape: list,
                 n_z: int,
                 **kwargs: dict):
        super().__init__(**kwargs)

        self.relu_fn = tf.nn.swish

        if have_condition:
            h, w, c = input_shape
            self.condition = tf.keras.layers.Dense(h*w, activation=self.relu_fn)
        else:
            self.condition = None

        self.layers = []
        for i, num in enumerate(encoder_hidden_dims):
            l = EncoderBlock(num, name='enc{}_{}'.format(i, num))
            self.layers.append(l)

        self.flatten = tf.keras.layers.Flatten()

        self.z_mu = tf.keras.layers.Dense(n_z)
        self.z_sigma = tf.keras.layers.Dense(n_z)


    def call(self, inputs: tf.Tensor, condition: tf.Tensor, training: bool):
        if self.condition:
            b, h, w, c = inputs.shape

            x = self.condition(condition, training=training)
            x = tf.reshape(x, [b, h, w, 1])
            x = tf.concat([inputs, x], axis=3)
        else:
            x = inputs

        for layer in self.layers:
            x = layer(x, training=training)

        x = self.flatten(x)

        mu = self.z_mu(x)

        # this is actually log(sigma**2)
        sigma = self.z_sigma(x)

        return mu, sigma

class Decoder(tf.keras.layers.Layer):
    def __init__(self,
                 decoder_hidden_dims: list,
                 have_condition: bool,
                 decoder_input_shape: list,
                 output_channels: int,
                 **kwargs: dict):
        super().__init__(**kwargs)

        self.relu_fn = tf.nn.swish
        self.have_condition = have_condition

        self.decoder_input_shape = decoder_input_shape
        h, w, c = decoder_input_shape
        self.embeddings = tf.keras.layers.Dense(h*w*c, name='embeddings', activation=self.relu_fn)

        self.layers = []
        for i, num in enumerate(decoder_hidden_dims):
            layer = DecoderBlock(num, name='dec{}_{}'.format(i, num))
            self.layers.append(layer)

        self.out = tf.keras.layers.Conv2DTranspose(output_channels, kernel_size=1, padding='same', activation='sigmoid')


    def call(self, z: tf.Tensor, condition: tf.Tensor, training: bool):
        if self.have_condition:
            x = tf.concat([z, condition], -1)
        else:
            x = z

        x = self.embeddings(x, training=training)
        x = tf.reshape(x, [-1] + self.decoder_input_shape)

        for layer in self.layers:
            x = layer(x, training=training)

        x = self.out(x, training=training)
        return x

class VAE(tf.keras.Model):
    def __init__(self,
                 have_condition: bool,
                 encoder_hidden_dims: list,
                 encoder_input_shape: list,
                 num_z: int,
                 decoder_hidden_dims: list,
                 decoder_input_shape: list,
                 decoder_output_channels: int,
                 **kwargs: dict):
        super().__init__(**kwargs)

        self.n_z = num_z

        h, w, _ = encoder_input_shape
        self.resize = tf.keras.layers.experimental.preprocessing.Resizing(h, w)
        self.rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1 / 255)

        self.encoder = Encoder(encoder_hidden_dims, have_condition, encoder_input_shape, num_z)
        self.decoder = Decoder(decoder_hidden_dims, have_condition, decoder_input_shape, decoder_output_channels)

    def sample_from_encoder_vars(self, mu, sigma):
        batch_size = tf.shape(mu)[0]
        eps = tf.random.normal([batch_size, self.n_z], mean=0, stddev=1)
        z = mu + tf.math.exp(sigma / 2) * eps

        return z

    def call(self, inputs: tf.Tensor, condition: tf.Tensor, training: bool):
        x = self.resize(inputs)
        scaled_x = self.rescale(x)

        mu, sigma = self.encoder(scaled_x, condition, training=training)

        z = self.sample_from_encoder_vars(mu, sigma)

        dec_output = self.decoder(z, condition, training)

        return scaled_x, dec_output, mu, sigma

    @tf.function
    def generate(self, condition: tf.Tensor, output_dir: str):
        batch_size = tf.shape(condition)[0]

        if True:
            z = tf.zeros([batch_size, self.n_z], dtype=tf.float32)
        else:
            z = tf.random.normal([batch_size, self.n_z], mean=0, stddev=1)

        dec_output = self.decoder(z, condition, training=False)
        grey_data = tf.cast(dec_output * 255, tf.uint8)

        indexes = tf.range(tf.shape(condition)[0])

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

        self.pixel_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
        #self.pixel_loss = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits, reduction=tf.keras.losses.Reduction.SUM)

    def reset_states(self):
        self.train_metric.reset_states()
        self.eval_metric.reset_states()

    def z_kl_loss_mc_sampling(self, mu, sigma):
        logpz = normal_log_pdf(z_sample, 0., 1.)  # shape=(batch_size,)
        logqz_x = normal_log_pdf(z_sample, mu, tf.math.square(sd))  # shape=(batch_size,)
        kl_divergence = logqz_x - logpz

    def z_kl_loss_analytical_for_gauss(self, mu, sigma):
        x = tf.math.exp(sigma) + tf.math.square(mu) - 1 - sigma
        x = 0.5 * tf.reduce_sum(x, -1)
        return x

    def __call__(self, true_images, pred_images, mu, sigma, training):
        rec_loss = self.pixel_loss(true_images, pred_images)
        #rec_loss = tf.reduce_sum(rec_loss, [1])

        kl_loss = self.z_kl_loss_analytical_for_gauss(mu, sigma)

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

    x_train = tf.expand_dims(x_train, 3)
    x_test = tf.expand_dims(x_test, 3)

    num_labels = 10
    y_train = tf.one_hot(y_train, num_labels)
    y_test = tf.one_hot(y_test, num_labels)

    batch_size = 512

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size).cache()

    encoder_hidden_dims = [16, 32, 64, 128]
    encoder_input_shape = [28, 28, 1]
    num_latent_vars = 2

    decoder_hidden_dims = [32, 16]

    model = VAE(have_condition=True,
                encoder_hidden_dims=encoder_hidden_dims,
                encoder_input_shape=encoder_input_shape,
                num_z=num_latent_vars,
                decoder_hidden_dims=decoder_hidden_dims,
                decoder_input_shape=[7, 7, 32],
                decoder_output_channels=1)

    model(tf.ones([batch_size] + encoder_input_shape, dtype=tf.float32), condition=tf.zeros([batch_size, num_labels], dtype=tf.float32), training=True)
    model.summary()

    generate_labels = tf.range(num_labels)
    generate_labels = tf.one_hot(generate_labels, num_labels)
    output_dir = tf.Variable('results/start')
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

        output_dir.assign('results/{}'.format(epoch))
        model.generate(generate_labels, output_dir)

if __name__ == '__main__':
    main()
