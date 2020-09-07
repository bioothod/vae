import tensorflow as tf

from tensorflow.keras.utils import get_custom_objects

def get_flops(model):
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.compat.v1.profiler.profile(
        graph=tf.compat.v1.keras.backend.get_session().graph, run_meta=run_meta, cmd="op", options=opts
    )

    return flops.total_float_ops  # Prints the "flops" of the model.


class Mish(tf.keras.layers.Activation):
    """
    based on https://github.com/digantamisra98/Mish/blob/master/Mish/TFKeras/mish.py
    Mish Activation Function.
    """

    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = "Mish"


def mish(inputs):
    result = inputs * tf.math.tanh(tf.math.softplus(inputs))
    return result


class GroupConv2D(tf.keras.layers.Layer):
    """
    https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_py
    Different group can have a different kernel size.
    """

    def __init__(self,
                 num_filters: int,
                 kernel_sizes: list,
                 **kwargs):
        super().__init__()

        self.channel_axis = -1

        splits = self.split_channels(num_filters, len(kernel_sizes))

        self.convs = []
        for num_group_features, group_kernel_size in zip(splits, kernel_sizes):
            conv = tf.keras.layers.Conv2D(num_group_features, kernel_size=group_kernel_size, **kwargs)
            self.convs.append(conv)

    def split_channels(self, num_filters, num_groups):
        num_filters_in_the_group = num_filters // num_groups
        round_diff = num_filters - num_filters_in_the_group * num_groups

        splits = [num_filters_in_the_group] * num_groups
        splits[0] += round_diff
        return splits

    def __call__(self, inputs):
        if len(self.convs) == 1:
            return self.convs[0](inputs)

        num_features = inputs.shape[self.channel_axis]
        splits = self.split_channels(num_features, len(self.convs))

        x = tf.split(inputs, splits, self.channel_axis)
        outputs = [conv(inp) for inp, conv in zip(x, self.convs)]
        outputs = tf.concat(outputs, self.channel_axis)
        return outputs


class ResNest:
    def __init__(self,
                 verbose=False,
                 input_shape=(224, 224, 3),
                 activation="relu",
                 include_top=True,
                 num_classes=81,
                 dropout_rate=0.2,
                 fc_activation=None,
                 blocks_set=[3, 4, 6, 3],
                 radix=2,
                 groups=1,
                 bottleneck_width=64,
                 deep_stem=True,
                 stem_width=32,
                 block_expansion=4,
                 avg_down=True,
                 avd=True,
                 avd_first=False,
                 preact=False,
                 using_basic_block=False,
                 using_cb=False,
                 **kwargs):

        self.channel_axis = -1
        self.verbose = verbose
        self.activation = activation
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.fc_activation = fc_activation

        self.blocks_set = blocks_set
        self.radix = radix
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width

        self.deep_stem = deep_stem
        self.stem_width = stem_width
        self.block_expansion = block_expansion
        self.avg_down = avg_down
        self.avd = avd
        self.avd_first = avd_first

        self.dilation = 1
        self.preact = preact
        self.using_basic_block = using_basic_block
        self.using_cb = using_cb

        self.include_top = include_top

    def make_stem(self, input_tensor, stem_width=64, deep_stem=False):
        x = input_tensor
        if deep_stem:
            x = tf.keras.layers.Conv2D(stem_width, kernel_size=3, strides=2, padding="same",
                                       kernel_initializer="he_normal", use_bias=False, data_format="channels_last")(x)

            x = tf.keras.layers.BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
            x = tf.keras.layers.Activation(self.activation)(x)

            x = tf.keras.layers.Conv2D(stem_width, kernel_size=3, strides=1, padding="same",
                                       kernel_initializer="he_normal", use_bias=False, data_format="channels_last")(x)

            x = tf.keras.layers.BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
            x = tf.keras.layers.Activation(self.activation)(x)

            x = tf.keras.layers.Conv2D(stem_width*2, kernel_size=3, strides=1, padding="same",
                                       kernel_initializer="he_normal", use_bias=False, data_format="channels_last")(x)

            #x = tf.keras.layers.BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
            #x = tf.keras.layers.Activation(self.activation)(x)
        else:
            x = tf.keras.layers.Conv2D(stem_width, kernel_size=7, strides=2, padding="same",
                                       kernel_initializer="he_normal", use_bias=False, data_format="channels_last")(x)
            #x = tf.keras.layers.BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
            #x = tf.keras.layers.Activation(self.activation)(x)
        return x

    def rsoftmax(self, input_tensor, filters, radix, groups):
        x = input_tensor
        if radix > 1:
            x = tf.reshape(x, [-1, groups, radix, filters//groups])
            x = tf.transpose(x, [0, 2, 1, 3])
            x = tf.keras.activations.softmax(x, axis=1)
            x = tf.reshape(x, [-1, 1, 1, radix*filters])
        else:
            x = tf.keras.layers.Activation("sigmoid")(x)
        return x

    def SplAtConv2d(self, input_tensor, filters=64, kernel_size=3, stride=1, dilation=1, groups=1, radix=0):
        x = input_tensor
        in_channels = input_tensor.shape[-1]

        x = GroupConv2D(num_filters=filters*radix, kernel_sizes=[kernel_size for i in range(groups*radix)],
                        padding="same", kernel_initializer="he_normal", use_bias=False,
                        data_format="channels_last", dilation_rate=dilation)(x)

        x = tf.keras.layers.BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
        x = tf.keras.layers.Activation(self.activation)(x)

        if radix > 1:
            splited = tf.split(x, radix, axis=-1)
            gap = sum(splited)
        else:
            gap = x

        # print('sum',gap.shape)
        gap = tf.keras.layers.GlobalAveragePooling2D(data_format="channels_last")(gap)
        gap = tf.reshape(gap, [-1, 1, 1, filters])
        # print('adaptive_avg_pool2d',gap.shape)

        reduction_factor = 4
        inter_channels = max((in_channels*radix) // reduction_factor, 32)

        x = tf.keras.layers.Conv2D(inter_channels, kernel_size=1)(gap)

        x = tf.keras.layers.BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
        x = tf.keras.layers.Activation(self.activation)(x)
        x = tf.keras.layers.Conv2D(filters*radix, kernel_size=1)(x)

        attention = self.rsoftmax(x, filters, radix, groups)

        if radix > 1:
            logits = tf.split(attention, radix, axis=-1)
            out = sum([a*b for a, b in zip(splited, logits)])
        else:
            out = attention * x
        return out

    def make_block(self,
                   input_tensor,
                   first_block=True,
                   filters=64,
                   stride=2,
                   radix=1,
                   avd=False,
                   avd_first=False,
                   is_first=False):
        x = input_tensor
        inplanes = input_tensor.shape[-1]
        shortcut = input_tensor
        if stride != 1 or inplanes != filters*self.block_expansion:
            if self.avg_down:
                if self.dilation == 1:
                    shortcut = tf.keras.layers.AveragePooling2D(pool_size=stride, strides=stride, padding="same", data_format="channels_last")(shortcut)
                else:
                    shortcut = tf.keras.layers.AveragePooling2D(pool_size=1, strides=1, padding="same", data_format="channels_last")(shortcut)

                shortcut = tf.keras.layers.Conv2D(filters*self.block_expansion, kernel_size=1, strides=1, padding="same",
                                                  kernel_initializer="he_normal", use_bias=False, data_format="channels_last")(shortcut)
            else:
                shortcut = tf.keras.layers.Conv2D(filters*self.block_expansion, kernel_size=1, strides=stride, padding="same",
                                                  kernel_initializer="he_normal", use_bias=False, data_format="channels_last")(shortcut)

            shortcut = tf.keras.layers.BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(shortcut)

        group_width = int(filters * (self.bottleneck_width / 64.0)) * self.cardinality
        x = tf.keras.layers.Conv2D(group_width, kernel_size=1, strides=1, padding="same", kernel_initializer="he_normal", use_bias=False, data_format="channels_last")(x)
        x = tf.keras.layers.BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
        x = tf.keras.layers.Activation(self.activation)(x)

        avd = avd and (stride > 1 or is_first)
        if avd:
            avd_layer = tf.keras.layers.AveragePooling2D(pool_size=3, strides=stride, padding="same", data_format="channels_last")
            stride = 1

        if avd and avd_first:
            x = avd_layer(x)

        if radix >= 1:
            x = self.SplAtConv2d(x, filters=group_width, kernel_size=3, stride=stride, dilation=self.dilation,
                                groups=self.cardinality, radix=radix)
        else:
            x = tf.keras.layers.Conv2D(group_width, kernel_size=3, strides=stride, padding="same", kernel_initializer="he_normal",
                                       dilation_rate=self.dilation, use_bias=False, data_format="channels_last")(x)
            x = tf.keras.layers.BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
            x = tf.keras.layers.Activation(self.activation)(x)

        if avd and not avd_first:
            x = avd_layer(x)

        x = tf.keras.layers.Conv2D(filters*self.block_expansion, kernel_size=1, strides=1, padding="same", kernel_initializer="he_normal",
                                   dilation_rate=self.dilation, use_bias=False, data_format="channels_last")(x)
        x = tf.keras.layers.BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)

        m2 = tf.keras.layers.Add()([x, shortcut])
        m2 = tf.keras.layers.Activation(self.activation)(m2)
        return m2

    def make_block_basic(self, input_tensor, first_block=True, filters=64, stride=2, radix=1, avd=False, avd_first=False, is_first=False):
        """Conv2d_BN_Relu->Bn_Relu_Conv2d
        """
        x = input_tensor
        x = tf.keras.layers.BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
        x = tf.keras.layers.Activation(self.activation)(x)

        shortcut = x
        inplanes = input_tensor.shape[-1]
        if stride != 1 or inplanes != filters*self.block_expansion:
            if self.avg_down:
                if self.dilation == 1:
                    shortcut = tf.keras.layers.AveragePooling2D(pool_size=stride, strides=stride, padding="same", data_format="channels_last")(shortcut)
                else:
                    shortcut = tf.keras.layers.AveragePooling2D(pool_size=1, strides=1, padding="same", data_format="channels_last")(shortcut)

                shortcut = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=1, padding="same", kernel_initializer="he_normal",
                                                  use_bias=False, data_format="channels_last")(shortcut)
            else:
                shortcut = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=stride, padding="same", kernel_initializer="he_normal",
                                                  use_bias=False, data_format="channels_last")(shortcut)

        group_width = int(filters * (self.bottleneck_width / 64.0)) * self.cardinality
        avd = avd and (stride > 1 or is_first)
        if avd:
            avd_layer = tf.keras.layers.AveragePooling2D(pool_size=3, strides=stride, padding="same", data_format="channels_last")
            stride = 1

        if avd and avd_first:
            x = avd_layer(x)

        if radix >= 1:
            x = self.SplAtConv2d(x, filters=group_width, kernel_size=3, stride=stride, dilation=self.dilation, groups=self.cardinality, radix=radix)
        else:
            x = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=stride, padding="same", kernel_initializer="he_normal",
                                       dilation_rate=self.dilation, use_bias=False, data_format="channels_last")(x)

        if avd and not avd_first:
            x = avd_layer(x)

        x = tf.keras.layers.BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
        x = tf.keras.layers.Activation(self.activation)(x)
        x = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding="same", kernel_initializer="he_normal",
                                   dilation_rate=self.dilation, use_bias=False, data_format="channels_last")(x)
        m2 = tf.keras.layers.Add()([x, shortcut])
        return m2

    def make_layer(self, input_tensor, blocks=4, filters=64, stride=2, is_first=True):
        x = input_tensor
        if self.using_basic_block is True:
            x = self.make_block_basic(x, first_block=True, filters=filters, stride=stride, radix=self.radix,
                                      avd=self.avd, avd_first=self.avd_first, is_first=is_first)

            for i in range(1, blocks):
                x = self.make_block_basic(x, first_block=False, filters=filters, stride=1, radix=self.radix, avd=self.avd, avd_first=self.avd_first)

        elif self.using_basic_block is False:
            x = self.make_block(x, first_block=True, filters=filters, stride=stride, radix=self.radix, avd=self.avd, avd_first=self.avd_first, is_first=is_first)

            for i in range(1, blocks):
                x = self.make_block(x, first_block=False, filters=filters, stride=1, radix=self.radix, avd=self.avd, avd_first=self.avd_first)

        return x

    def make_composite_layer(self, input_tensor, filters=256, kernel_size=1, stride=1, upsample=True):
        x = input_tensor
        x = tf.keras.layers.Conv2D(filters, kernel_size, strides=stride, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
        if upsample:
            x = tf.keras.layers.UpSampling2D(size=2)(x)
        return x

    def build(self):
        get_custom_objects().update({'mish': Mish(mish)})

        input_sig = tf.keras.Input(shape=self.input_shape)
        x = self.make_stem(input_sig, stem_width=self.stem_width, deep_stem=self.deep_stem)

        if self.preact is False:
            x = tf.keras.layers.BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
            x = tf.keras.layers.Activation(self.activation)(x)
        if self.verbose:
            print("stem_out", x.shape)

        x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same", data_format="channels_last")(x)
        if self.verbose:
            print("MaxPool2D out", x.shape)

        if self.preact is True:
            x = tf.keras.layers.BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
            x = tf.keras.layers.Activation(self.activation)(x)
        
        if self.using_cb:
            second_x = x
            second_x = self.make_layer(x, blocks=self.blocks_set[0], filters=64, stride=1, is_first=False)
            second_x_tmp = self.make_composite_layer(second_x, filters=x.shape[-1], upsample=False)
            if self.verbose:
                print('layer 0 db_com', second_x_tmp.shape)

            x = tf.keras.layers.Add()([second_x_tmp, x])

        x = self.make_layer(x, blocks=self.blocks_set[0], filters=64, stride=1, is_first=False)
        if self.verbose:
            print("-" * 5, "layer 0 out", x.shape, "-" * 5)

        b1_b3_filters = [64, 128, 256, 512]
        for i in range(3):
            idx = i + 1

            if self.using_cb:
                second_x = self.make_layer(x, blocks=self.blocks_set[idx], filters=b1_b3_filters[idx], stride=2)
                second_x_tmp = self.make_composite_layer(second_x, filters=x.shape[-1])
                if self.verbose:
                    print('layer {} db_com out {}'.format(idx, second_x_tmp.shape))
                x = tf.keras.layers.Add()([second_x_tmp, x])

            x = self.make_layer(x, blocks=self.blocks_set[idx], filters=b1_b3_filters[idx], stride=2)
            if self.verbose:
                print('----- layer {} out {} -----'.format(idx, x.shape))

        if self.include_top:
            x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
            if self.verbose:
                print("pool_out:", x.shape)

            if self.dropout_rate > 0:
                x = tf.keras.layers.Dropout(self.dropout_rate, noise_shape=None)(x)

            fc_out = tf.keras.layers.Dense(self.n_classes, kernel_initializer="he_normal", use_bias=False, name="fc_NObias")(x)
            if self.verbose:
                print("fc_out:", fc_out.shape)

            if self.fc_activation:
                fc_out = tf.keras.layers.Activation(self.fc_activation)(fc_out)

            model = tf.keras.Model(inputs=input_sig, outputs=fc_out)

            if self.verbose:
                print("Resnest builded with input {}, output{}".format(input_sig.shape, fc_out.shape))
                print("-------------------------------------------")
                print("")
        else:
            model = tf.keras.Model(inputs=input_sig, outputs=x)

        return model
