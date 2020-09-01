import logging

import tensorflow as tf

logger = logging.getLogger("cifar")

relu_fn = tf.nn.relu
global_bn_eps = 1.001e-5

class BaseV2(tf.keras.Model):
    def __init__(self, **kwargs):
        super(BaseV2, self).__init__(**kwargs)

        self.pad0 = tf.keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='pad0')
        self.conv0 = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, use_bias=True, name='conv0')

        self.pad1 = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pad1')
        self.max_pooling = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, name='max_pooling')

    def call(self, inputs, training):
        x = self.pad0(inputs)
        x = self.conv0(x)

        x = self.pad1(x)
        x = self.max_pooling(x)

        return x

class SmallBlockV2(tf.keras.Model):
    def __init__(self, channels_in, kernel_size=3, stride=1, conv_shortcut=False, **kwargs):
        super().__init__(**kwargs)

        self.preact_bn = tf.keras.layers.BatchNormalization(epsilon=global_bn_eps, name='preact_bn')

        self.pad0 = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pad0')
        self.bn0 = tf.keras.layers.BatchNormalization(epsilon=global_bn_eps, name='bn0')
        self.conv0 = tf.keras.layers.Conv2D(channels_in, kernel_size=3, strides=1, use_bias=False, name='conv0')

        self.pad1 = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pad1')
        self.conv1 = tf.keras.layers.Conv2D(channels_in, kernel_size=kernel_size, strides=stride, use_bias=False, name='conv1')

        if conv_shortcut:
            self.shortcut = tf.keras.layers.Conv2D(channels_in, kernel_size=1, strides=stride, name='shortcut_conv2d')
        else:
            if stride > 1:
                self.shortcut = tf.keras.layers.MaxPool2D(pool_size=1, strides=stride, name='shortcut_max_pooling')
            else:
                self.shortcut = lambda x: x

    def call(self, inputs, training):
        x = self.preact_bn(inputs, training=training)
        preact = relu_fn(x)

        shortcut = self.shortcut(preact)

        x = self.pad0(preact)
        x = self.conv0(x)
        x = self.bn0(x, training=training)
        x = relu_fn(x)

        x = self.pad1(x)
        x = self.conv1(x)

        x += shortcut
        return x

class BlockV2(tf.keras.Model):
    def __init__(self, channels_in, kernel_size=3, stride=1, conv_shortcut=False, **kwargs):
        super(BlockV2, self).__init__(**kwargs)

        channels_out = channels_in * 4
        self.preact_bn = tf.keras.layers.BatchNormalization(epsilon=global_bn_eps, name='preact_bn')

        if conv_shortcut:
            self.shortcut = tf.keras.layers.Conv2D(channels_out, kernel_size=1, strides=stride, name='shortcut_conv2d')
        else:
            if stride > 1:
                self.shortcut = tf.keras.layers.MaxPool2D(pool_size=1, strides=stride, name='shortcut_max_pooling')
            else:
                self.shortcut = lambda x: x

        self.conv0 = tf.keras.layers.Conv2D(channels_in, kernel_size=1, strides=1, use_bias=False, name='conv0')
        self.bn0 = tf.keras.layers.BatchNormalization(epsilon=global_bn_eps, name='bn0')

        self.pad1 = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pad1')
        self.conv1 = tf.keras.layers.Conv2D(channels_in, kernel_size=kernel_size, strides=stride, use_bias=False, name='conv1')
        self.bn1 = tf.keras.layers.BatchNormalization(epsilon=global_bn_eps, name='bn1')

        self.conv2 = tf.keras.layers.Conv2D(channels_out, kernel_size=1, strides=1, name='conv2')

    def call(self, inputs, training):
        x = self.preact_bn(inputs, training=training)
        preact = relu_fn(x)

        shortcut = self.shortcut(preact)

        x = self.conv0(preact)
        x = self.bn0(x, training=training)
        x = relu_fn(x)

        x = self.pad1(x)
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = relu_fn(x)

        x = self.conv2(x)
        x += shortcut
        return x

class StackV2(tf.keras.Model):
    def __init__(self, channels_in, num_blocks, block=BlockV2, stride1=2, **kwargs):
        super(StackV2, self).__init__(**kwargs)

        self.blocks = [block(channels_in, conv_shortcut=True, name='block0')]

        for i in range(2, num_blocks):
            self.blocks.append(block(channels_in, name='block{}'.format(i-1)))

        self.blocks.append(block(channels_in, stride=stride1, name='block{}'.format(num_blocks-1)))

    def call(self, x, training):
        for block in self.blocks:
            x = block(x, training=training)

        return x

class ResNet18V2(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.base = BaseV2()
        self.stacks = []
        self.stacks.append(StackV2(64, num_blocks=2, block=SmallBlockV2, name='stack0'))
        self.stacks.append(StackV2(128, num_blocks=2, block=SmallBlockV2, name='stack1'))
        self.stacks.append(StackV2(256, num_blocks=2, block=SmallBlockV2, name='stack2'))
        self.stacks.append(StackV2(512, num_blocks=2, block=SmallBlockV2, stride1=1, name='stack3'))

        self.post_bn = tf.keras.layers.BatchNormalization(epsilon=global_bn_eps, name='post_bn')

    def call(self, inputs, training):
        x = self.base(inputs, training=training)

        for stack in self.stacks:
            x = stack(x, training=training)

        x = self.post_bn(x, training=training)
        x = relu_fn(x)

        return x

class ResNet34V2(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.base = BaseV2()
        self.stacks = []
        self.stacks.append(StackV2(64, num_blocks=3, block=SmallBlockV2, name='stack0'))
        self.stacks.append(StackV2(128, num_blocks=4, block=SmallBlockV2, name='stack1'))
        self.stacks.append(StackV2(256, num_blocks=6, block=SmallBlockV2, name='stack2'))
        self.stacks.append(StackV2(512, num_blocks=3, block=SmallBlockV2, stride1=1, name='stack3'))

        self.post_bn = tf.keras.layers.BatchNormalization(epsilon=global_bn_eps, name='post_bn')

    def call(self, inputs, training):
        x = self.base(inputs, training=training)

        for stack in self.stacks:
            x = stack(x, training=training)

        x = self.post_bn(x, training=training)
        x = relu_fn(x)

        return x

class ResNet50V2(tf.keras.Model):
    def __init__(self, **kwargs):
        super(ResNet50V2, self).__init__(**kwargs)

        self.base = BaseV2()
        self.stacks = []
        self.stacks.append(StackV2(64, num_blocks=3, name='stack0'))
        self.stacks.append(StackV2(128, num_blocks=4, name='stack1'))
        self.stacks.append(StackV2(256, num_blocks=6, name='stack2'))
        self.stacks.append(StackV2(512, num_blocks=3, stride1=1, name='stack3'))

        self.post_bn = tf.keras.layers.BatchNormalization(epsilon=global_bn_eps, name='post_bn')

    def call(self, inputs, training):
        x = self.base(inputs, training=training)

        for stack in self.stacks:
            x = stack(x, training=training)

        x = self.post_bn(x, training=training)
        x = relu_fn(x)

        return x

class ResNet101V2(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.base = BaseV2()
        self.stacks = []
        self.stacks.append(StackV2(64, num_blocks=3, name='stack0'))
        self.stacks.append(StackV2(128, num_blocks=4, name='stack1'))
        self.stacks.append(StackV2(256, num_blocks=23, name='stack2'))
        self.stacks.append(StackV2(512, num_blocks=3, stride1=1, name='stack3'))

        self.post_bn = tf.keras.layers.BatchNormalization(epsilon=global_bn_eps, name='post_bn')

    def call(self, inputs, training):
        x = self.base(inputs, training=training)

        for stack in self.stacks:
            x = stack(x, training=training)

        x = self.post_bn(x, training=training)
        x = relu_fn(x)

        return x

class ResNet152V2(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.base = BaseV2()
        self.stacks = []
        self.stacks.append(StackV2(64, num_blocks=3, name='stack0'))
        self.stacks.append(StackV2(128, num_blocks=8, name='stack1'))
        self.stacks.append(StackV2(256, num_blocks=36, name='stack2'))
        self.stacks.append(StackV2(512, num_blocks=3, stride1=1, name='stack3'))

        self.post_bn = tf.keras.layers.BatchNormalization(epsilon=global_bn_eps, name='post_bn')

    def call(self, inputs, training):
        x = self.base(inputs, training=training)

        for stack in self.stacks:
            x = stack(x, training=training)

        x = self.post_bn(x, training=training)
        x = relu_fn(x)

        return x
