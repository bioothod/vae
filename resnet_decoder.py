import tensorflow as tf

class BaseV2(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.conv = tf.keras.layers.Conv2DTranspose(3, kernel_size=7, strides=2, use_bias=True, padding='same')
        self.upsampling = tf.keras.layers.UpSampling2D(size=2)

    def call(self, inputs, training):
        x = self.upsampling(inputs)
        x = self.conv(x)
        return x

class BlockV2(tf.keras.layers.Layer):
    def __init__(self, channels_in, kernel_size, stride=1, conv_shortcut=False, **kwargs):
        super().__init__(**kwargs)

        channels_out = channels_in // 4
        self.preact_bn = tf.keras.layers.BatchNormalization()

        self.bn0 = tf.keras.layers.BatchNormalization()
        self.conv0 = tf.keras.layers.Conv2DTranspose(channels_in, kernel_size=1, strides=1, use_bias=False, padding='same')

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2DTranspose(channels_out, kernel_size=kernel_size, strides=stride, use_bias=False, padding='same')

        self.conv2 = tf.keras.layers.Conv2DTranspose(channels_out, kernel_size=1, stride=1, padding='same')

        if conv_shortcut:
            self.shortcut = tf.keras.layers.Conv2DTranspose(channels_out, kernel_size=1, strides=stride, padding='same')
        else:
            if stride > 1:
                self.shortcut = tf.keras.layers.UpSampling2D(size=stride)
            else:
                self.shortcut = lambda x: x

    def call(self, inputs, training):
        x = self.preact_bn(inputs, training=training)
        preact = relu_fn(x)

        shortcut = self.shortcut(preact)

        x = self.conv0(preact)
        x = self.bn0(x, training=training)
        x = relu_fn(x)

        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = relu_fn(x)

        x = self.conv2(x)
        x += shortcut
        return x

class StackV2(tf.keras.layers.Layer):
    def __init__(self, channels_in, blocks, stride1=2, **kwargs):
        super().__init__(**kwargs)

        self.blocks = [BlockV2(channels_in, conv_shortcut=True, name='block1')]
        for i in range(2, blocks):
            self.blocks.append(BlockV2(channels_in, name='block{}'.format(i)))

        self.blocks.append(BlockV2(channels_in, stride=stride1, name='block{}'.format(blocks)))

    def call(self, x, training):
        for block in self.blocks:
            x = block(x, training=training)

        return x


class ResNet50V2Decoder(tf.keras.layers.Layer):
    def __init__(self):
        self.stacks = []
        self.stacks.append(StackV2(512, blocks=3, stride1=1, name='conv5'))
        self.stacks.append(StackV2(256, blocks=6, name='conv4'))
        self.stacks.append(StackV2(128, blocks=4, name='conv3'))
        self.stacks.append(StackV2(64, blocks=3, name='conv2'))

        self.base = BaseV2()

        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training):
        for stack in self.stacks:
            x = stack(x, training=training)

        x = self.base(x, training=training)
        return x

