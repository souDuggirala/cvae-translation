import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
from tensorflow.keras.models import Model
#import numpy as np
#from scipy.special import softmax
from tensorflow.keras import backend as K

'''
returns model whose
    input: source and target annotation vectors
        shape of annotation vector tensor: (batch_size, timestep (words/vectors), hidden_dim)
        !might have to set an upper limit on the number of words, but also maybe not needed
    output: mean and logvar from posterior
'''
def infer_posterior(hidden_dim, latent_dim):
    
    #dimension is twice hidden because of concatenation
    s_ann_vectors = Input(shape = (None, hidden_dim))
    t_ann_vectors = Input(shape = (None, hidden_dim))
    
    #lambda layer which takes in annotation vectors and return context vectors
    c_s = coattention(hidden_dim)([s_ann_vectors, t_ann_vectors])
    c_t = coattention(hidden_dim)([t_ann_vectors, s_ann_vectors])

    #mean pooling
        #take mean of annotation and context vectors
    c_s_mean = layers.Lambda(meanpool)(c_s)
    c_t_mean = layers.Lambda(meanpool)(c_t)
    s_ann_vectors_mean = layers.Lambda(meanpool)(s_ann_vectors)
    t_ann_vectors_mean = layers.Lambda(meanpool)(t_ann_vectors)
    
    concatenated = layers.Concatenate()([c_s_mean, c_t_mean, s_ann_vectors_mean, t_ann_vectors_mean])

    projection = layers.Dense(latent_dim, activation = 'tanh')(concatenated)
    #no activation for mean and variance projection
    mean = layers.Dense(latent_dim) (projection)
    logvar = layers.Dense(latent_dim) (projection)#vector with describes the diagonal of the covariance matrix
    
    model = Model([s_ann_vectors, t_ann_vectors], [mean, logvar], name = 'infer_posterior') #!and more if needed for loss
    
    return model

'''
returns model whose
    input: annotation vectors
    output: tensor of context vectors that correspond with the first tensor of annotation vectors inputted
        i.e., if calculating c_x, input ann_vectors_x first

from paper:
alpha_x_t = softmax(h_y_t dot h_x^T)
alpha_y_t = softmax(h_x_t dot h_y^T)
c_x_t = alpha_x_t dot h_x
c_y_t = alpha_y_t dot h_y
(x = source, y = target, h = tensor of annotation vectors, t = timestep)
'''
def coattention(hidden_dim):
    
    
    ann_vectors_1 = Input(shape = (None, hidden_dim))
    ann_vectors_2 = Input(shape = (None, hidden_dim))
    
    #(batch_size, num_vectors, hidden_dim) dot (batch_size, num_vectors, hidden_dim)
    #axes param (2,2): dot axis 2 of first with axis 2 of second, equivalent to transposing the second one
    dot_products = layers.Dot(axes = (2,2))([ann_vectors_2, ann_vectors_1])
    #take softmax of each row of matrix product
    alphas = layers.Softmax(axis = -1)(dot_products)
    context_vectors = layers.Dot(axes = (2,1))([alphas, ann_vectors_1])
    
    model = Model([ann_vectors_1, ann_vectors_2], context_vectors)
    
    return model

'''
returns model whose...
    input: source and target annotation vectors
    output: mean and logvar from prior
'''
def infer_prior(hidden_dim, latent_dim):
    
    s_ann_vectors = Input(shape = (None, hidden_dim))
    s_ann_vectors_mean = layers.Lambda(meanpool)(s_ann_vectors)
    
    projection = layers.Dense(latent_dim, activation = 'tanh')(s_ann_vectors_mean)
    #no activation for mean and variance projection
    mean = layers.Dense(latent_dim) (projection)
    logvar = layers.Dense(latent_dim) (projection) #vector with describes the diagonal of the covariance matrix
    
    model = Model(s_ann_vectors, [mean, logvar], name = 'infer_prior') #!and more if needed for loss
    
    return model

'''
Given 3D tensor of shape (batch_size, num_vecs, vec_len), take mean over vectors to get (batch_size, vec_len)
!There's probably a keras layer that does this
'''
def meanpool(args):
    tensor = args
    return K.mean(tensor, axis = 1)

'''
samples z from gaussian given mean and log variance
call() copied from Keras book Lambda example
subclasses to use the training argument
'''
class Sampler(layers.Layer):
    def __init__(self):
        super(Sampler, self).__init__()
        
    def call(self, inputs):
        '''
        if training:
            mean = inputs[0]
            log_var = inputs[1]
        else:          
            mean = inputs[2]
            log_var = inputs[3]
        '''#changed my mind, I'm going to build the translator from the pieces of the model
        mean = inputs[0]
        log_var = inputs[1]
            
        #find out parameters to make sure shapes are right
        epsilon = K.random_normal(shape = K.shape(mean), mean = 0, stddev = 1)
        
        return mean + K.exp(log_var)*epsilon

    