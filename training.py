import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
import os
import pickle
from model import build_translation_model
import dataset

'''
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
tf.config.experimental.set_memory_growth(physical_devices[1], enable=True)
tf.config.experimental.set_memory_growth(physical_devices[2], enable=True)
tf.config.experimental.set_memory_growth(physical_devices[3], enable=True)'''

#code repurposed from https://deepakbaby.in/post/vae-insights/ and tensorflow seq2seq NMT tutorial 
#read more about custom losses from https://towardsdatascience.com/how-to-create-a-custom-loss-function-keras-3a89156ec69b
#also used tensorflow guide https://www.tensorflow.org/guide/keras/train_and_evaluate

# total number of epochs
n_epochs = 500 
# The number of epochs at which KL loss should be included
klstart = 40
# number of epochs over which KL scaling is increased from 0 to 1
kl_annealtime = 20

class AnnealingCallback(Callback):
    def __init__(self, weight):
        self.weight = weight
    def on_epoch_end (self, epoch, logs={}):
        if epoch > klstart :
            new_weight = min(K.get_value(self.weight) + (1./ annealtime), 1.)
            K.set_value(self.weight, new_weight)
        print ("Current KL Weight is " + str(K.get_value(self.weight)))


'''
def vae_loss(weight):
   
    #Loss = - reconstruction + kl loss
    #look at custom losses link for explaination of parameters
    # target_sentence shape = (BATCH_SIZE, max_length_output)
    # logits shape = (BATCH_SIZE, max_length_output, t_vocab_size )
    def loss (target_sentence, logits):
    
'''
        ##Old reconstruction loss
        #getting probabilities for each vocab word could be too computationally intensive
            #https://blog.csdn.net/linli522362242/article/details/115518150
        #this loss could be counting the padding, which don't want to do
        #misunderstanding: the rnn_output are not the probabilities, only the output of the Dense layer
            #Dense layer does not do softmax, sampler does
        #old parameters:
            #"translation" (sample_ids)
            #"word_prob_distr" (rnn_output) now "logits"
'''
        #get word_probs from word_prob_distr and translation
            #translation shape (batch_size, len_of_sentence) - each word is an index
            #word_prob_distr shape (batch_size, len_of_sentence, vocab_size)
            #one hot creates a mask vector (vocab_size large) that has '1' at the word index
        mask = K.one_hot(translation, word_prob_distr.shape[2]) 
        word_probs = tf.math.multiply(mask, word_prob_distr)
        word_probs = K.sum(word_probs, axis = 2)

        #reconstruction loss for whole batch= - (1/batch_size) sum (log p(y |x, z))
        #given word_probs (shape = (batch_size, len_of_sentence)), log and sum over axis 1 is log p(y |x, z)
        #reconstruction_loss = - K.sum(K.log(word_probs))/word_probs.shape[0] #!output needs to be shape (batch_size,)
        reconstruction_loss = - K.sum(K.log(word_probs), axis = 1)
'''
        #Newer reconstruction loss
'''
        #instead use loss from tensorflow seq2seq NMT tutorial
        #cut off <start> token from target_sentence (assumes post padding)
        target_sentence = target_sentence[:, 1:]
        #cross entropy loss
        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        reconstruction_loss = cross_entropy(y_true=target_sentence, y_pred=logits)
        #mask to ignore padding (returns 0 when y=0 else 1)
        mask = tf.logical_not(tf.math.equal(target_sentence,0))
        mask = tf.cast(mask, dtype=reconstruction_loss.dtype)  
        reconstruction_loss = mask* reconstruction_loss
        reconstruction_loss = tf.reduce_mean(reconstruction_loss)
'''
        #kl loss
'''        
        # kl loss
        # not the same as in standard VAE implementations because of prior
        kl_loss = - prior_logvar + post_logvar + 1 - K.exp(prior_logvar)/K.exp(post_logvar) - K.square(prior_mean - post_mean)/K.exp(prior_logvar) #!possible divide by 0 issue
        kl_loss = K.sum(kl_loss, axis=-1) #sum along latent dim
        kl_loss *= - 0.5 #because the formula has a 1/2 in front from the gaussian pdf
        
        
        return reconstruction_loss #+ (weight * kl_loss)

    return loss
'''        
        

def get_compiled_model(s_words, t_words, weight):
    model = build_translation_model(s_words, t_words)
    model.compile(
        optimizer='adam',
        loss = None,
        loss_weights = [1, weight]
        #loss=vae_loss(weight)
    )
    return model

def make_or_restore_model(s_words, t_words, weight, checkpoint_dir):
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)
    print("Creating a new model")
    return get_compiled_model(s_words, t_words, weight)


#Train the model

dirname = 'jesc_prepared'
raw_dir = dirname + '/raw'
train_dir = dirname + '/train'
val_dir = dirname + '/dev'
tokenizer_dir = 'tokenizers'

tokenizer_s, tokenizer_t = dataset.get_tokenizers(raw_dir, tokenizer_dir)
s_words = tokenizer_s.word_index
t_words = tokenizer_t.word_index
s_train, t_train = dataset.get_sequences_from_dir(train_dir, tokenizer_s, tokenizer_t)
s_val, t_val = dataset.get_sequences_from_dir(val_dir, tokenizer_s, tokenizer_t)

# Prepare a directory to store all the checkpoints.
checkpoint_dir = "./ckpt"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    
#AnnealingCallback will print out the latest weight, so when restarting, change weight to that val
#or get is from history?
weight = K.variable(0.)
callbacks = [
    AnnealingCallback(weight),
    # This callback saves a SavedModel every 100 batches.
    # We include the training loss in the saved model name.
    ModelCheckpoint(
        filepath=checkpoint_dir + "/epoch={epoch}_ckpt-loss={loss:.2f}", save_freq=100
    )
]
'''
keras.callbacks.ModelCheckpoint(
    # Path where to save the model
    # The two parameters below mean that we will overwrite
    # the current checkpoint if and only if
    # the `val_loss` score has improved.
    # The saved model name will include the current epoch.
    filepath="mymodel_{epoch}",
    save_best_only=True,  # Only save a model if `val_loss` has improved.
    monitor="val_loss",
    verbose=1,
'''

model = make_or_restore_model(s_words, t_words, weight, checkpoint_dir)
history = model.fit(x = [s_train, t_train], y = None, epochs=1, callbacks = callbacks, validation_data = ([s_val, t_val], None))

with open('history.pickle', 'wb') as f:
    pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)

    
'''Resources/Prior Research'''
#cross entropy for reconstruction loss:
    #https://stats.stackexchange.com/questions/288451/why-is-mean-squared-error-the-cross-entropy-between-the-empirical-distribution-a/288453
    #https://stats.stackexchange.com/questions/323568/help-understanding-reconstruction-loss-in-variational-autoencoder
    #answer found in cross entropy wikipedia page section "Estimation"
        #don't calculate integral, just do maximum likelihood estimation

#derivation of KL loss from code in apendix of Kingma & Welling (2014)
    #mine will have to be different because I have a mean and logvar for prior and standard VAE assumes 0 mean and identity covariance matrix
    #calculate closed form solution for KL loss, then use that formula with mean and logvar from both prior and posterior
    
#how to train VAE in Keras: https://deepakbaby.in/post/vae-insights/, https://tiao.io/post/tutorial-on-variational-autoencoders-with-a-concise-keras-implementation/

