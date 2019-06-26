from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from .instance_normalization import InstanceNormalization
from .GroupNormalization import GroupNormalization
from .pixel_shuffler import PixelShuffler
from .custom_layers.scale_layer import Scale
from .custom_inits.icnr_initializer import icnr_keras
import tensorflow as tf
import keras.backend as K
from .layers import PixelShuffler, Scale, SubPixelUpscaling, ReflectionPadding2D
from keras.initializers import he_uniform, Constant
#from .normalization import GroupNormalization, InstanceNormalization
# initializers and weight decay regularization are fixed
conv_init = 'he_normal'
w_l2 = 1e-4

class ICNR(initializers.Initializer):  # pylint: disable=invalid-name
    '''
    ICNR initializer for checkerboard artifact free sub pixel convolution

    Andrew Aitken et al. Checkerboard artifact free sub-pixel convolution
    https://arxiv.org/pdf/1707.02937.pdf	https://distill.pub/2016/deconv-checkerboard/

    Parameters:
        initializer: initializer used for sub kernels (orthogonal, glorot uniform, etc.)
        scale: scale factor of sub pixel convolution (upsampling from 8x8 to 16x16 is scale 2)
    Return:
        The modified kernel weights
    Example:
        x = conv2d(... weights_initializer=ICNR(initializer=he_uniform(), scale=2))
    '''

    def __init__(self, initializer, scale=2):
        self.scale = scale
        self.initializer = initializer

    def __call__(self, shape, dtype='float32'):  # tf needs partition_info=None
        shape = list(shape)
        if self.scale == 1:
            return self.initializer(shape)
        new_shape = shape[:3] + [shape[3] // (self.scale ** 2)]
        if type(self.initializer) is dict:
            self.initializer = initializers.deserialize(self.initializer)
        var_x = self.initializer(new_shape, dtype)
        var_x = tf.transpose(var_x, perm=[2, 0, 1, 3])
        var_x = tf.image.resize_nearest_neighbor(
                         var_x,
                         size=(shape[0] * self.scale, shape[1] * self.scale),
                         align_corners=True)
        var_x = tf.space_to_depth(var_x, block_size=self.scale, data_format='NHWC')
        var_x = tf.transpose(var_x, perm=[1, 2, 0, 3])
        return var_x

    def get_config(self):
        config = {'scale': self.scale,
                  'initializer': self.initializer
                  }
        base_config = super(ICNR, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def update_kwargs(kwargs):
    """ Set the default kernel initializer to he_uniform() """
    kwargs["kernel_initializer"] = kwargs.get("kernel_initializer", he_uniform())
    return kwargs
# <<< Original Model Blocks >>> #
def conv( inp, filters, kernel_size=5, strides=2, padding='same', use_instance_norm=True, res_block_follows=True,
         **kwargs):
    """ Convolution Layer"""

    inp = ReflectionPadding2D(stride=strides, kernel_size=kernel_size)(inp)
    padding = 'valid'
    var_x = Conv2D(filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding=padding,
                   **kwargs)(inp)
    if use_instance_norm:
        var_x = InstanceNormalization()(var_x)
    if not res_block_follows:
        var_x = LeakyReLU(0.1)(var_x)
    return var_x


def upscale(inp, filters, kernel_size=3, padding='same', use_instance_norm=True, res_block_follows=True,
            **kwargs):
    kwargs = update_kwargs(kwargs)
    inp = ReflectionPadding2D(stride=1, kernel_size=kernel_size)(inp)
    padding = 'valid'

    kwargs["kernel_initializer"]= ICNR(initializer=kwargs["kernel_initializer"])
    var_x = Conv2D(filters * 4,
                   kernel_size=kernel_size,
                   padding=padding,
                   **kwargs)(inp)
    if use_instance_norm:
        var_x = InstanceNormalization()(var_x)
    if not res_block_follows:
        var_x = LeakyReLU(0.1)(var_x)
    #if self.use_subpixel:
    var_x = SubPixelUpscaling()(var_x)
    #else:
    #    var_x = PixelShuffler()(var_x)
    return var_x


# <<< DFaker Model Blocks >>> #
def new_res_block(inp, filters, kernel_size=3, padding='same', **kwargs):
    """ Residual block """

    kwargs = update_kwargs(kwargs)
    var_x = LeakyReLU(alpha=0.2)(inp)

    var_x = ReflectionPadding2D(stride=1, kernel_size=kernel_size)(var_x)
    padding = 'valid'
    var_x = Conv2D(filters,
                   kernel_size=kernel_size,
                   padding=padding,
                   **kwargs)(var_x)
    var_x = LeakyReLU(alpha=0.2)(var_x)

    var_x = ReflectionPadding2D(stride=1, kernel_size=kernel_size)(var_x)
    padding = 'valid'
    var_x = Conv2D(filters,
                   kernel_size=kernel_size,
                   padding=padding,
                   **kwargs)(var_x)
    var_x = Scale(gamma_init=Constant(value=0.1))(var_x)
    var_x = Add()([var_x, inp])
    var_x = LeakyReLU(alpha=0.2)(var_x)
    return var_x


# <<< Unbalanced Model Blocks >>> #
def conv_sep(inp, filters, kernel_size=5, strides=2, **kwargs):
    """ Seperable Convolution Layer """

    kwargs = update_kwargs(kwargs)
    var_x = SeparableConv2D(filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding='same',
                            **kwargs)(inp)
    var_x = Activation("relu")(var_x)
    return var_x


def self_attn_block(inp, nc, squeeze_factor=8):
    '''
    Code borrows from https://github.com/taki0112/Self-Attention-GAN-Tensorflow
    '''
    assert nc//squeeze_factor > 0, f"Input channels must be >= {squeeze_factor}, recieved nc={nc}"
    x = inp
    shape_x = x.get_shape().as_list()
    
    f = Conv2D(nc//squeeze_factor, 1, kernel_regularizer=regularizers.l2(w_l2))(x)
    g = Conv2D(nc//squeeze_factor, 1, kernel_regularizer=regularizers.l2(w_l2))(x)
    h = Conv2D(nc, 1, kernel_regularizer=regularizers.l2(w_l2))(x)
    
    shape_f = f.get_shape().as_list()
    shape_g = g.get_shape().as_list()
    shape_h = h.get_shape().as_list()
    flat_f = Reshape((-1, shape_f[-1]))(f)
    flat_g = Reshape((-1, shape_g[-1]))(g)
    flat_h = Reshape((-1, shape_h[-1]))(h)   
    
    s = Lambda(lambda x: K.batch_dot(x[0], Permute((2,1))(x[1])))([flat_g, flat_f])

    beta = Softmax(axis=-1)(s)
    o = Lambda(lambda x: K.batch_dot(x[0], x[1]))([beta, flat_h])
    o = Reshape(shape_x[1:])(o)
    o = Scale()(o)
    
    out = add([o, inp])
    return out

def dual_attn_block(inp, nc, squeeze_factor=8):
    '''
    https://github.com/junfu1115/DANet
    '''
    assert nc//squeeze_factor > 0, f"Input channels must be >= {squeeze_factor}, recieved nc={nc}"
    x = inp
    shape_x = x.get_shape().as_list()
    
    # position attention module
    x_pam = Conv2D(nc, kernel_size=3, kernel_regularizer=regularizers.l2(w_l2),  
               kernel_initializer=conv_init, use_bias=False, padding="same")(x)
    x_pam = Activation("relu")(x_pam)
    x_pam = normalization(x_pam, norm, nc)
    f_pam = Conv2D(nc//squeeze_factor, 1, kernel_regularizer=regularizers.l2(w_l2))(x_pam)
    g_pam = Conv2D(nc//squeeze_factor, 1, kernel_regularizer=regularizers.l2(w_l2))(x_pam)
    h_pam = Conv2D(nc, 1, kernel_regularizer=regularizers.l2(w_l2))(x_pam)    
    shape_f_pam = f_pam.get_shape().as_list()
    shape_g_pam = g_pam.get_shape().as_list()
    shape_h_pam = h_pam.get_shape().as_list()
    flat_f_pam = Reshape((-1, shape_f_pam[-1]))(f_pam)
    flat_g_pam = Reshape((-1, shape_g_pam[-1]))(g_pam)
    flat_h_pam = Reshape((-1, shape_h_pam[-1]))(h_pam)    
    s_pam = Lambda(lambda x: K.batch_dot(x[0], Permute((2,1))(x[1])))([flat_g_pam, flat_f_pam])
    beta_pam = Softmax(axis=-1)(s_pam)
    o_pam = Lambda(lambda x: K.batch_dot(x[0], x[1]))([beta_pam, flat_h_pam])
    o_pam = Reshape(shape_x[1:])(o_pam)
    o_pam = Scale()(o_pam)    
    out_pam = add([o_pam, x_pam])
    out_pam = Conv2D(nc, kernel_size=3, kernel_regularizer=regularizers.l2(w_l2),  
               kernel_initializer=conv_init, use_bias=False, padding="same")(out_pam)
    out_pam = Activation("relu")(out_pam)
    out_pam = normalization(out_pam, norm, nc)
    
    # channel attention module
    x_chn = Conv2D(nc, kernel_size=3, kernel_regularizer=regularizers.l2(w_l2),  
               kernel_initializer=conv_init, use_bias=False, padding="same")(x)
    x_chn = Activation("relu")(x_chn)
    x_chn = normalization(x_chn, norm, nc)
    shape_x_chn = x_chn.get_shape().as_list()
    flat_f_chn = Reshape((-1, shape_x_chn[-1]))(x_chn)
    flat_g_chn = Reshape((-1, shape_x_chn[-1]))(x_chn)
    flat_h_chn = Reshape((-1, shape_x_chn[-1]))(x_chn)    
    s_chn = Lambda(lambda x: K.batch_dot(Permute((2,1))(x[0]), x[1]))([flat_g_chn, flat_f_chn])
    s_new_chn = Lambda(lambda x: K.repeat_elements(K.max(x, -1, keepdims=True), nc, -1))(s_chn)
    s_new_chn = Lambda(lambda x: x[0] - x[1])([s_new_chn, s_chn])
    beta_chn = Softmax(axis=-1)(s_new_chn)
    o_chn = Lambda(lambda x: K.batch_dot(x[0], Permute((2,1))(x[1])))([flat_h_chn, beta_chn])
    o_chn = Reshape(shape_x[1:])(o_chn)
    o_chn = Scale()(o_chn)    
    out_chn = add([o_chn, x_chn])
    out_chn = Conv2D(nc, kernel_size=3, kernel_regularizer=regularizers.l2(w_l2),  
               kernel_initializer=conv_init, use_bias=False, padding="same")(out_chn)
    out_chn = Activation("relu")(out_chn)
    out_chn = normalization(out_chn, norm, nc)
    
    out = add([out_pam, out_chn])
    return out

def normalization(inp, norm='none', group='16'):    
    x = inp
    if norm == 'layernorm':
        x = GroupNormalization(group=group)(x)
    elif norm == 'batchnorm':
        x = BatchNormalization()(x)
    elif norm == 'groupnorm':
        x = GroupNormalization(group=16)(x)
    elif norm == 'instancenorm':
        x = InstanceNormalization()(x)
    elif norm == 'hybrid':
        if group % 2 == 1:
            raise ValueError(f"Output channels must be an even number for hybrid norm, received {group}.")
        f = group
        x0 = Lambda(lambda x: x[...,:f//2])(x)
        x1 = Lambda(lambda x: x[...,f//2:])(x)        
        x0 = Conv2D(f//2, kernel_size=1, kernel_regularizer=regularizers.l2(w_l2),
                    kernel_initializer=conv_init)(x0)
        x1 = InstanceNormalization()(x1)        
        x = concatenate([x0, x1], axis=-1)
    else:
        x = x
    return x

def conv_block(input_tensor, f, use_norm=False, strides=2, w_l2=w_l2, norm='none'):
    x = input_tensor
    x = Conv2D(f, kernel_size=3, strides=strides, kernel_regularizer=regularizers.l2(w_l2),  
               kernel_initializer=conv_init, use_bias=False, padding="same")(x)
    x = Activation("relu")(x)
    x = normalization(x, norm, f) if use_norm else x
    return x

def conv_block_d(input_tensor, f, use_norm=False, w_l2=w_l2, norm='none'):
    x = input_tensor
    x = Conv2D(f, kernel_size=4, strides=2, kernel_regularizer=regularizers.l2(w_l2), 
               kernel_initializer=conv_init, use_bias=False, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)   
    x = normalization(x, norm, f) if use_norm else x
    return x

def res_block(input_tensor, f, use_norm=False, w_l2=w_l2, norm='none'):
    x = input_tensor
    x = Conv2D(f, kernel_size=3, kernel_regularizer=regularizers.l2(w_l2), 
               kernel_initializer=conv_init, use_bias=False, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = normalization(x, norm, f) if use_norm else x
    x = Conv2D(f, kernel_size=3, kernel_regularizer=regularizers.l2(w_l2), 
               kernel_initializer=conv_init, use_bias=False, padding="same")(x)
    x = add([x, input_tensor])
    x = LeakyReLU(alpha=0.2)(x)
    x = normalization(x, norm, f) if use_norm else x
    return x

def SPADE_res_block(input_tensor, cond_input_tensor, f, use_norm=True, norm='none'):
    """
    Semantic Image Synthesis with Spatially-Adaptive Normalization
    Taesung Park, Ming-Yu Liu, Ting-Chun Wang, Jun-Yan Zhu
    https://arxiv.org/abs/1903.07291

    Note:
        SPADE just works like a charm. 
        It speeds up training alot and is also a very promosing approach for solving profile face generation issue.
        *(This implementation can be wrong since I haven't finished reading the paper. 
          The author hasn't release their code either (https://github.com/NVlabs/SPADE).)
    """
    def SPADE(input_tensor, cond_input_tensor, f, use_norm=True, norm='none'):
        x = input_tensor
        x = normalization(x, norm, f) if use_norm else x
        y = cond_input_tensor
        y = Conv2D(128, kernel_size=3, kernel_regularizer=regularizers.l2(w_l2), 
                   kernel_initializer=conv_init, padding='same')(y)
        y = Activation('relu')(y)           
        gamma = Conv2D(f, kernel_size=3, kernel_regularizer=regularizers.l2(w_l2), 
                   kernel_initializer=conv_init, padding='same')(y)
        beta = Conv2D(f, kernel_size=3, kernel_regularizer=regularizers.l2(w_l2), 
                   kernel_initializer=conv_init, padding='same')(y)
        x = add([x, multiply([x, gamma])])
        x = add([x, beta])
        return x
        
    x = input_tensor
    x = SPADE(x, cond_input_tensor, f, use_norm, norm)
    x = Activation('relu')(x)
    x = ReflectPadding2D(x)
    x = Conv2D(f, kernel_size=3, kernel_regularizer=regularizers.l2(w_l2), 
               kernel_initializer=conv_init, use_bias=not use_norm)(x)
    x = SPADE(x, cond_input_tensor, f, use_norm, norm)
    x = Activation('relu')(x)
    x = ReflectPadding2D(x)
    x = Conv2D(f, kernel_size=3, kernel_regularizer=regularizers.l2(w_l2), 
               kernel_initializer=conv_init)(x)
    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def upscale_ps(input_tensor, f, use_norm=False, w_l2=w_l2, norm='none'):
    x = input_tensor
    x = Conv2D(f*4, kernel_size=3, kernel_regularizer=regularizers.l2(w_l2), 
               kernel_initializer=icnr_keras, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = normalization(x, norm, f) if use_norm else x
    x = PixelShuffler()(x)
    return x

def ReflectPadding2D(x, pad=1):
    x = Lambda(lambda x: tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT'))(x)
    return x

def upscale_nn(input_tensor, f, use_norm=False, w_l2=w_l2, norm='none'):
    x = input_tensor
    x = UpSampling2D()(x)
    x = ReflectPadding2D(x, 1)
    x = Conv2D(f, kernel_size=3, kernel_regularizer=regularizers.l2(w_l2), 
               kernel_initializer=conv_init)(x)
    x = normalization(x, norm, f) if use_norm else x
    return x
