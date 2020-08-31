import argparse
import logging
import os
import re

import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

import horovod.tensorflow as hvd

import resnet

logger = logging.getLogger("cifar")

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128, help='Number of images to process in a batch')
parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs to run')
parser.add_argument('--train_dir', type=str, required=True, help='Path to train directory, where graph will be stored')
parser.add_argument('--model_name', type=str, default='', help='Model name')
parser.add_argument('--data_format', type=str, default='channels_last', choices=['channels_first', 'channels_last'], help='Data format: [channels_first, channels_last]')
parser.add_argument('--initial_learning_rate', default=1e-3, type=float, help='Initial learning rate (will be multiplied by the number of nodes in the distributed strategy)')
parser.add_argument('--min_learning_rate', default=1e-6, type=float, help='Minimal learning rate')
parser.add_argument('--print_per_train_steps', default=20, type=int, help='Print train stats per this number of steps(batches)')
parser.add_argument('--min_eval_metric', default=0.2, type=float, help='Minimal evaluation metric to start saving models')
parser.add_argument('--epochs_lr_update', default=20, type=int, help='Maximum number of epochs without improvement used to reset or decrease learning rate')
parser.add_argument('--use_fp16', action='store_true', help='Whether to use fp16 training/inference')
parser.add_argument('--steps_per_train_epoch', default=-1, type=int, help='Number of steps per train run')
parser.add_argument('--steps_per_eval_epoch', default=-1, type=int, help='Number of steps per evaluation run')
parser.add_argument('--reset_on_lr_update', action='store_true', help='Whether to reset to the best model after learning rate update')
parser.add_argument('--rotation_augmentation', type=float, default=0, help='Rotation augmentation angle, value <= 0 disables it')
parser.add_argument('--reg_loss_weight', type=float, default=0, help='L2 regularization weight')
parser.add_argument('--image_size', type=int, default=224, help='Image size')
FLAGS = parser.parse_args()

class Metric:
    def __init__(self, training):
        self.reg_loss = tf.keras.metrics.Mean(name='reg_loss')
        self.ce_loss = tf.keras.metrics.Mean(name='ce_loss')

        self.total_loss = tf.keras.metrics.Mean(name='total_loss')

        self.acc = tf.keras.metrics.Mean(name='ce_acc')

        self.training = training

    def reset_states(self):
        self.reg_loss.reset_states()
        self.ce_loss.reset_states()
        self.total_loss.reset_states()

        self.acc.reset_states()

    def str_result(self):
        if self.training:
            return 'loss: ce: {:.4e}, reg_loss: {:.4e}, total: {:.4e}, accuracy: {:.4f}'.format(
                    self.ce_loss.result(),
                    self.reg_loss.result(),
                    self.total_loss.result(),
                    self.acc.result())
        else:
            return 'loss: ce: {:.4e}, accuracy: {:.4f}'.format(self.ce_loss.result(), self.acc.result())

class MetricAggregator:
    def __init__(self, from_logits: bool):
        self.train_metric = Metric(training=True)
        self.eval_metric = Metric(training=False)

        self.ce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=from_logits, reduction=tf.keras.losses.Reduction.AUTO)

    def reset_states(self):
        self.train_metric.reset_states()
        self.eval_metric.reset_states()

    def str_result(self, training):
        m = self.train_metric if training else self.eval_metric
        return m.str_result()

    def evaluation_result(self):
        return self.eval_metric.acc.result()

    def __call__(self, y_true, y_pred, training):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        ce_loss = self.ce_loss(y_true, y_pred)

        m = self.train_metric if training else self.eval_metric
        m.ce_loss.update_state(ce_loss)

        m.acc.update_state(y_true, y_pred)

        return ce_loss

class Model(tf.keras.Model):
    def __init__(self, model_name, num_classes, image_size, classifier_activation='softmax', dtype=tf.float32, **kwargs):
        super(Model, self).__init__()

        self.model_dtype = dtype

        #self.resize = tf.keras.layers.experimental.preprocessing.Resizing(image_size, image_size)
        #self.random_flip = tf.keras.layers.experimental.preprocessing.RandomFlip(mode='horizontal')
        #self.rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1 / 255)

        if model_name == 'resnet50v2':
            self.features = resnet.ResNet50V2()
        elif model_name == 'resnet101v2':
            self.features = resnet.ResNet101V2()
        elif model_name == 'resnet101v2_keras':
            self.features = tf.keras.applications.ResNet101V2(include_top=False, weights=None)
        elif model_name == 'resnet50v2_keras':
            self.features = tf.keras.applications.ResNet50V2(include_top=False, weights=None)
        else:
            raise ValueError('model name "{}" is not supported'.format(model_name))

        self.avg_pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.dense = tf.keras.layers.Dense(num_classes, activation=classifier_activation)

        inputs = tf.keras.Input([None, None, 3])
        outputs = self.call(inputs, training=True)
        super(Model, self).__init__(inputs=[inputs], outputs=[outputs])

    def call(self, inputs, training):
        #x = self.resize(inputs)
        #x = tf.cast(x, self.model_dtype)

        #x = self.random_flip(x)
        #x = self.rescale(x)

        #x = self.features(x, training)
        x = self.features(inputs, training)

        x = self.avg_pooling(x)
        x = self.dense(x)

        return x

def pad_resize_image(image, dims):
    image = tf.image.resize(image, dims, preserve_aspect_ratio=True)

    shape = tf.shape(image)

    sxd = dims[1] - shape[1]
    syd = dims[0] - shape[0]

    sx = tf.cast(sxd / 2, dtype=tf.int32)
    sy = tf.cast(syd / 2, dtype=tf.int32)

    paddings = tf.convert_to_tensor([[sy, syd - sy], [sx, sxd - sx], [0, 0]])
    image = tf.pad(image, paddings, mode='CONSTANT', constant_values=128)
    return image

def tf_read_image(image, label, dtype):
    #image = tf.io.read_file(filename)
    #image = tf.io.decode_jpeg(image, channels=3)

    image = tf.image.resize(image, [FLAGS.image_size, FLAGS.image_size], preserve_aspect_ratio=False)
    image = tf.cast(image, dtype)
    image = image / 255

    image = tf.image.random_flip_left_right(image)

    return image, label

def main():
    hvd.init()

    logger.propagate = False
    logger.setLevel(logging.INFO)
    __fmt = logging.Formatter(fmt='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%d/%m/%y %H:%M:%S')
    if hvd.rank() == 0:
        __handler = logging.StreamHandler()
        __handler.setFormatter(__fmt)
        logger.addHandler(__handler)

    checkpoint_dir = os.path.join(FLAGS.train_dir, 'checkpoints')
    good_checkpoint_dir = os.path.join(checkpoint_dir, 'good')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(good_checkpoint_dir, exist_ok=True)

    handler = logging.FileHandler(os.path.join(checkpoint_dir, 'train.log.{}'.format(hvd.rank())), 'a')
    handler.setFormatter(__fmt)
    logger.addHandler(handler)

    gpus = tf.config.experimental.list_physical_devices('GPU')

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    num_replicas = hvd.size()

    dtype = tf.float32
    if FLAGS.use_fp16:
        dtype = tf.float16
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)

    FLAGS.initial_learning_rate *= num_replicas

    if hvd.rank() == 0:
        logdir = os.path.join(FLAGS.train_dir, 'logs')
        writer = tf.summary.create_file_writer(logdir)
        writer.set_as_default()

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    num_train_images = len(x_train)
    num_eval_images = len(x_test)

    num_classes = 10
    y_train = tf.one_hot(y_train, num_classes)
    y_test = tf.one_hot(y_test, num_classes)

    y_train = tf.squeeze(y_train, 1)
    y_test = tf.squeeze(y_test, 1)

    
    def create_dataset(name, images, labels, training):
        ds = tf.data.Dataset.from_tensor_slices((images, labels))

        if training:
            ds = ds.shuffle(10000)

        ds = ds.map(lambda image, label: tf_read_image(image, label, dtype),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)

        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        ds = ds.batch(FLAGS.batch_size)

        if not training:
            ds = ds.cache()

        logging.info('{}: {}: dataset has been created'.format(hvd.rank(), name))

        return ds

    train_ds = create_dataset('train', x_train, y_train, training=True)
    eval_ds = create_dataset('eval', x_test, y_test, training=False)

    model = Model(model_name=FLAGS.model_name, num_classes=num_classes, image_size=FLAGS.image_size, dtype=dtype)

    metric = MetricAggregator(from_logits=False)

    global_step = tf.Variable(0, dtype=tf.int64, name='global_step', aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
    learning_rate = tf.Variable(FLAGS.initial_learning_rate, dtype=tf.float32, name='learning_rate')
    epoch_var = tf.Variable(0, dtype=tf.float32, name='epoch_number', aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

    opt = tf.keras.optimizers.Adam(lr=learning_rate)
    if FLAGS.use_fp16:
        opt = mixed_precision.LossScaleOptimizer(opt, loss_scale='dynamic')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            y_pred = model(images, training=True)
            total_loss = metric(labels, y_pred, training=True)

            if FLAGS.reg_loss_weight != 0:
                regex = r'.*(kernel|weight):0$'
                var_match = re.compile(regex)

                l2_loss = FLAGS.reg_loss_weight * tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables if var_match.match(v.name)])
                metric.train_metric.reg_loss.update_state(l2_loss)

                total_loss += l2_loss

            if FLAGS.use_fp16:
                scaled_total_loss = opt.get_scaled_loss(total_loss)

        metric.train_metric.total_loss.update_state(total_loss)

        if FLAGS.use_fp16:
            scaled_gradients = tape.gradient(scaled_total_loss, model.trainable_variables)
            gradients = opt.get_unscaled_gradients(scaled_gradients)
        else:
            gradients = tape.gradient(total_loss, model.trainable_variables)

        opt.apply_gradients(zip(gradients, model.trainable_variables))

        return total_loss

    @tf.function
    def eval_step(images, labels):
        y_pred = model(images, training=False)
        total_loss = metric(labels, y_pred, training=False)
        metric.eval_metric.total_loss.update_state(total_loss)
        return total_loss

    def run_epoch(name, dataset, step_func, max_steps, broadcast_variables=False):
        if name == 'train':
            m = metric.train_metric
        else:
            m = metric.eval_metric

        step = 0
        def log_progress():
            if name == 'train':
                logger.info('{}: step: {} {}/{}: {}'.format(
                    int(epoch_var.numpy()), global_step.numpy(), step, max_steps,
                    metric.str_result(True),
                    ))

            if hvd.rank() == 0:
                if name == 'train':
                    tf.summary.scalar('{}/lr'.format(name), learning_rate, step=global_step)
                    tf.summary.scalar('{}/epoch'.format(name), epoch, step=global_step)

                    tf.summary.scalar('{}/total_loss'.format(name), m.total_loss.result(), step=global_step)
                    tf.summary.scalar('{}/reg_loss'.format(name), m.reg_loss.result(), step=global_step)

                tf.summary.scalar('{}/ce_loss'.format(name), m.ce_loss.result(), step=global_step)
                tf.summary.scalar('{}/accuracy'.format(name), m.acc.result(), step=global_step)

        first_batch = True
        for images, labels in dataset:
            total_loss = step_func(images, labels)

            if name == 'train':
                if first_batch and broadcast_variables:
                    logger.info('broadcasting initial variables')
                    hvd.broadcast_variables(model.variables, root_rank=0)
                    hvd.broadcast_variables(opt.variables(), root_rank=0)
                    first_batch = False

                if (step % FLAGS.print_per_train_steps == 0) or np.isnan(total_loss.numpy()):
                    log_progress()

                    if np.isnan(total_loss.numpy()):
                        exit(-1)


            step += 1
            global_step.assign_add(1)
            if step >= max_steps:
                break

        log_progress()

        return step

    def calc_epoch_steps(num_images):
        return (num_images + FLAGS.batch_size - 1) // (FLAGS.batch_size)

    steps_per_train_epoch = FLAGS.steps_per_train_epoch
    if steps_per_train_epoch < 0:
        steps_per_train_epoch = calc_epoch_steps(num_train_images)

    steps_per_eval_epoch = FLAGS.steps_per_eval_epoch
    if steps_per_eval_epoch < 0:
        steps_per_eval_epoch = calc_epoch_steps(num_eval_images)

    logger.info('model_name: {}, image_size: {}, num_classes: {}, steps_per_train_epoch: {}, steps_per_eval_epoch: {}, train_images: {}, eval_images: {}'.format(
        FLAGS.model_name, FLAGS.image_size, num_classes,
        steps_per_train_epoch, steps_per_eval_epoch,
        num_train_images, num_eval_images))

    best_metric = 0
    best_saved_path = None
    num_epochs_without_improvement = 0
    initial_learning_rate_multiplier = 0.2
    learning_rate_multiplier = initial_learning_rate_multiplier

    def validation_metric():
        return metric.evaluation_result()

    #model(tf.ones([FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3], dtype=tf.uint8), training=True)
    model(tf.ones([FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3], dtype=dtype), training=True)

    def log_layer(m, spaces=0):
        for l in m.layers:
            name = l.name + ','
            output_shapes = [l.get_output_shape_at(i) for i in range(len(l._inbound_nodes))]
            output_shapes = str(output_shapes) + ','

            num_params = np.sum([np.prod(v.shape) for v in l.trainable_variables])
            num_params = int(num_params)
            logger.info('{} {:24s} output_shapes: {:56s} num_params: {}'.format(' '*spaces, name, output_shapes, num_params))
            if hasattr(l, 'layers'):
                spaces += 1
                log_layer(l, spaces)
                spaces -= 1


    log_layer(model)

    num_vars = len(model.trainable_variables)
    num_params = np.sum([np.prod(v.shape) for v in model.trainable_variables])

    logger.info('nodes: {}, checkpoint_dir: {}, model: {}, image_size: {}, model trainable variables/params: {}/{}'.format(
        num_replicas, checkpoint_dir, FLAGS.model_name, FLAGS.image_size,
        num_vars, int(num_params)))

    def validation_metric():
        return metric.evaluation_result()

    learning_rate.assign(FLAGS.initial_learning_rate)
    for epoch in range(FLAGS.num_epochs):
        metric.reset_states()
        want_reset = False

        train_steps = run_epoch('train', train_ds, train_step, steps_per_train_epoch, (epoch == 0))
        eval_steps = run_epoch('eval', eval_ds, eval_step, steps_per_eval_epoch, broadcast_variables=False)

        new_lr = learning_rate.numpy()
        new_metric = validation_metric()

        logger.info('epoch: {}/{}, train: steps: {}, lr: {:.2e}, train: {}, eval: {}, metric: {:.4f}/{:.4f}'.format(
            int(epoch_var.numpy()), num_epochs_without_improvement, global_step.numpy(),
            learning_rate.numpy(),
            metric.str_result(True), metric.str_result(False),
            new_metric, best_metric))

        epoch_var.assign_add(1)

        if new_metric > best_metric:
            logger.info("epoch: {}/{}, global_step: {}, eval metric: {:.4f} -> {:.4f}: {}".format(
                int(epoch_var.numpy()), num_epochs_without_improvement, global_step.numpy(), best_metric, new_metric, metric.str_result(False)))

            best_metric = new_metric

            num_epochs_without_improvement = 0
            learning_rate_multiplier = initial_learning_rate_multiplier
        else:
            num_epochs_without_improvement += 1


        if num_epochs_without_improvement >= FLAGS.epochs_lr_update:
            if learning_rate > FLAGS.min_learning_rate:
                new_lr = learning_rate.numpy() * learning_rate_multiplier
                if new_lr < FLAGS.min_learning_rate:
                    new_lr = FLAGS.min_learning_rate

                if FLAGS.reset_on_lr_update:
                    want_reset = True

                logger.info('epoch: {}/{}, global_step: {}, epochs without metric improvement: {}, best metric: {:.5f}, updating learning rate: {:.2e} -> {:.2e}, will reset: {}'.format(
                    int(epoch_var.numpy()), num_epochs_without_improvement, global_step.numpy(), num_epochs_without_improvement, best_metric, learning_rate.numpy(), new_lr, want_reset))
                num_epochs_without_improvement = 0
                if learning_rate_multiplier > 0.1:
                    learning_rate_multiplier /= 2


            elif num_epochs_without_improvement >= FLAGS.epochs_lr_update:
                new_lr = FLAGS.initial_learning_rate
                want_reset = True

                logger.info('epoch: {}/{}, global_step: {}, epochs without metric improvement: {}, best metric: {:.5f}, resetting learning rate: {:.2e} -> {:.2e}, will reset: {}'.format(
                    int(epoch_var.numpy()), num_epochs_without_improvement, global_step.numpy(), num_epochs_without_improvement, best_metric, learning_rate.numpy(), new_lr, want_reset))

                num_epochs_without_improvement = 0
                learning_rate_multiplier = initial_learning_rate_multiplier


        # update learning rate even without resetting model
        learning_rate.assign(new_lr)

if __name__ == '__main__':
    main()
