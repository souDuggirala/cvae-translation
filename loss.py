import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

class KLLossLayer(layers.Layer):
    
    def kl_loss(self, post_mean, post_logvar, prior_mean, prior_logvar):
        
        # not the same as in standard VAE implementations because of prior
        kl_loss = - prior_logvar + post_logvar + 1 - K.exp(prior_logvar)/K.exp(post_logvar) - K.square(prior_mean - post_mean)/K.exp(prior_logvar) #!possible divide by 0 issue
        kl_loss = K.sum(kl_loss, axis=-1) #sum along latent dim
        kl_loss *= - 0.5 #because the formula has a 1/2 in front from the gaussian pdf
        
        return kl_loss
    
    def call(self, inputs):
        post_mean = inputs[0]
        post_logvar = inputs[1]
        prior_mean = inputs[2]
        prior_logvar = inputs[3]
        
        loss = self.kl_loss(post_mean, post_logvar, prior_mean, prior_logvar)
        self.add_loss(loss, inputs = inputs)
        return [post_mean, post_logvar, prior_mean, prior_logvar]
    
class ReconstructionLossLayer(layers.Layer):
    
    #from tensorflow seq2seq NMT tutorial
    def rec_loss(self, target_sentence, logits):
        
        target_sentence = target_sentence[:, 1:]
        paddings = tf.constant([[0,0],[0,1]])
        target_sentence = tf.pad(target_sentence, paddings)
        
        #print("Logits shape")
        #print(logits.shape)
        #print("Target Sentence shape")
        #print(target_sentence.shape)
        
        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        rec_loss = cross_entropy(y_true=target_sentence, y_pred=logits)
        #mask to ignore padding (returns 0 when y=0 else 1)
        mask = tf.logical_not(tf.math.equal(target_sentence,0))
        mask = tf.cast(mask, dtype=rec_loss.dtype)  
        rec_loss = mask* rec_loss
        rec_loss = tf.reduce_mean(rec_loss)
        
        return rec_loss
        
    def call(self, inputs):
        target_sentence = inputs[0]
        logits = inputs[1]
        
        loss = self.rec_loss(target_sentence, logits)
        self.add_loss(loss, inputs = inputs)
        return target_sentence
     