import tensorflow as tf

relu_fn = tf.nn.relu
global_bn_eps = 1.001e-5

class BaseV2(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.pad0 = tf.keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')
        self.conv0 = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, use_bias=True)

        self.pad1 = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')
        self.max_pooling0 = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)

    def call(self, inputs, training):
        x = self.pad0(inputs)
        x = self.conv0(x)

        x = self.pad1(x)
        x = self.max_pooling0(x)

        return x

class SmallBlockV2(tf.keras.layers.Layer):
    def __init__(self, channels_in, kernel_size=3, stride=1, conv_shortcut=False, **kwargs):
        super().__init__(**kwargs)

        self.preact_bn = tf.keras.layers.BatchNormalization(epsilon=global_bn_eps)

        self.bn0 = tf.keras.layers.BatchNormalization(epsilon=global_bn_eps)
        self.conv0 = tf.keras.layers.Conv2D(channels_in, kernel_size=3, strides=1, use_bias=False)

        self.pad1 = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv_1_pad')
        self.bn1 = tf.keras.layers.BatchNormalization(epsilon=global_bn_eps)
        self.conv1 = tf.keras.layers.Conv2D(channels_in, kernel_size=kernel_size, strides=stride, use_bias=False)

        if conv_shortcut:
            self.shortcut = tf.keras.layers.Conv2D(channels_in, kernel_size=1, strides=stride)
        else:
            if stride > 1:
                self.shortcut = tf.keras.layers.MaxPool2D(pool_size=1, strides=stride)
            else:
                self.shortcut = lambda x: x

    def call(self, inputs, training):
        x = self.preact_bn(inputs, training=training)
        preact = relu_fn(x)

        shortcut = self.shortcut(preact)

        x = self.conv0(preact)
        x = self.bn0(x, training=training)
        x = relu_fn(x)

        x = self.pad1(x)
        x = self.conv1(x)

        x += shortcut
        return x

class BlockV2(tf.keras.layers.Layer):
    def __init__(self, channels_in, kernel_size=3, stride=1, conv_shortcut=False, **kwargs):
        super().__init__(**kwargs)

        channels_out = channels_in * 4
        self.preact_bn = tf.keras.layers.BatchNormalization(epsilon=global_bn_eps)

        self.bn0 = tf.keras.layers.BatchNormalization(epsilon=global_bn_eps)
        self.conv0 = tf.keras.layers.Conv2D(channels_in, kernel_size=1, strides=1, use_bias=False, name='conv0')

        self.pad1 = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv_1_pad')
        self.bn1 = tf.keras.layers.BatchNormalization(epsilon=global_bn_eps)
        self.conv1 = tf.keras.layers.Conv2D(channels_in, kernel_size=kernel_size, strides=stride, use_bias=False)

        self.conv2 = tf.keras.layers.Conv2D(channels_out, kernel_size=1, strides=1)

        if conv_shortcut:
            self.shortcut = tf.keras.layers.Conv2D(channels_out, kernel_size=1, strides=stride, name='shortcut')
        else:
            if stride > 1:
                self.shortcut = tf.keras.layers.MaxPool2D(pool_size=1, strides=stride)
            else:
                self.shortcut = lambda x: x

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

class StackV2(tf.keras.layers.Layer):
    def __init__(self, channels_in, num_blocks, block=BlockV2, stride1=2, **kwargs):
        super().__init__(**kwargs)

        self.blocks = [block(channels_in, conv_shortcut=True, name='block0')]
        for i in range(2, num_blocks):
            self.blocks.append(block(channels_in, name='block{}'.format(i-1)))

        self.blocks.append(block(channels_in, stride=stride1, name='block{}'.format(num_blocks)))

    def call(self, x, training):
        for block in self.blocks:
            x = block(x, training=training)

        return x

class ResNet18V2(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.base = BaseV2()
        self.stacks = []
        self.stacks.append(StackV2(64, blocks=2, block=SmallBlockV2, name='stack0'))
        self.stacks.append(StackV2(128, blocks=2, block=SmallBlockV2,, name='stack1'))
        self.stacks.append(StackV2(256, blocks=2, block=SmallBlockV2,, name='stack2'))
        self.stacks.append(StackV2(512, blocks=2, block=SmallBlockV2,, stride1=1, name='stack3'))

    def call(self, inputs, training):
        x = self.base(inputs, training=training)

        for stack in self.stacks:
            x = stack(x, training=training)

        return x

class ResNet34V2(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.base = BaseV2()
        self.stacks = []
        self.stacks.append(StackV2(64, blocks=3, block=SmallBlockV2,, name='stack0'))
        self.stacks.append(StackV2(128, blocks=4, block=SmallBlockV2,, name='stack1'))
        self.stacks.append(StackV2(256, blocks=6, block=SmallBlockV2,, name='stack2'))
        self.stacks.append(StackV2(512, blocks=3, block=SmallBlockV2,, stride1=1, name='stack3'))

    def call(self, inputs, training):
        x = self.base(inputs, training=training)

        for stack in self.stacks:
            x = stack(x, training=training)

        return x

class ResNet50V2(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.base = BaseV2()
        self.stacks = []
        self.stacks.append(StackV2(64, blocks=3, name='conv2'))
        self.stacks.append(StackV2(128, blocks=4, name='conv3'))
        self.stacks.append(StackV2(256, blocks=6, name='conv4'))
        self.stacks.append(StackV2(512, blocks=3, stride1=1, name='conv5'))

    def call(self, inputs, training):
        x = self.base(inputs, training=training)

        for stack in self.stacks:
            x = stack(x, training=training)

        return x

class ResNet101V2(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.base = BaseV2()
        self.stacks = []
        self.stacks.append(StackV2(64, blocks=3, name='conv2'))
        self.stacks.append(StackV2(128, blocks=4, name='conv3'))
        self.stacks.append(StackV2(256, blocks=23, name='conv4'))
        self.stacks.append(StackV2(512, blocks=3, stride1=1, name='conv5'))

    def call(self, inputs, training):
        x = self.base(inputs, training=training)

        for stack in self.stacks:
            x = stack(x, training=training)

        return x

class ResNet152V2(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.base = BaseV2()
        self.stacks = []
        self.stacks.append(StackV2(64, blocks=3, name='conv2'))
        self.stacks.append(StackV2(128, blocks=8, name='conv3'))
        self.stacks.append(StackV2(256, blocks=36, name='conv4'))
        self.stacks.append(StackV2(512, blocks=3, stride1=1, name='conv5'))

    def call(self, inputs, training):
        x = self.base(inputs, training=training)

        for stack in self.stacks:
            x = stack(x, training=training)

        return x
