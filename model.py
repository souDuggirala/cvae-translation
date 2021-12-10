import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
#import tensorflow_addons as tfa
import inference, dataset, loss#, decoding_old, decoding_example
import numpy as np

EMBED_DIM = 300
HIDDEN_DIM = 300 #dimension of hidden states used for encoder and decoder LSTMS
LATENT_DIM = 32
#BATCH_SZ = 50
MAX_LEN_SOURCE = 50
MAX_LEN_TARGET = 50


def build_translation_model(s_words, t_words):

    s_vocab_size = len(s_words) + 1 #add 1 because word index does not include padding
    t_vocab_size = len(t_words) + 1
    
    #each sentence input is a list of word indices
    source_sentence = Input(shape = (MAX_LEN_SOURCE), name = 'source_input')
    target_sentence = Input(shape = (MAX_LEN_TARGET), name = 'target_input')
    
    #word embeddings (will reuse target_embedding)
    #another optional argument is max_length of sentence, which I don't think I'll need
    source_embedding = layers.Embedding(s_vocab_size, EMBED_DIM, name = 'source_embedding')
    target_embedding = layers.Embedding(t_vocab_size, EMBED_DIM, name = 'target_embedding') 
    s_embedded = source_embedding(source_sentence)
    t_embedded = target_embedding(target_sentence)
    
    #encoding with bidirectional lstm
    #annotation vectors will end up having dimension of HIDDEN_DIM*2 cuz of concat
    #return_sequences = True because want all the annotation vectors
    #return_states = True for source_encoder because need to set initial state of decoder
    source_encoder = layers.Bidirectional(layers.LSTM(HIDDEN_DIM, return_sequences = True, return_state = True), name = 'source_encoder') 
    target_encoder = layers.Bidirectional(layers.LSTM(HIDDEN_DIM, return_sequences = True), name = 'target_encoder')
    s_annotation_vectors, _, _, back_h, back_c = source_encoder(s_embedded)
    t_annotation_vectors = target_encoder(t_embedded)
    
    #infererence
    #both are here because need to train both
    infer_posterior = inference.infer_posterior(HIDDEN_DIM*2, LATENT_DIM)
    infer_prior = inference.infer_prior(HIDDEN_DIM*2, LATENT_DIM)
    post_mean, post_logvar = infer_posterior([s_annotation_vectors, t_annotation_vectors])
    prior_mean, prior_logvar = infer_prior(s_annotation_vectors)
    
    post_mean, post_logvar, prior_mean, prior_logvar = loss.KLLossLayer()([post_mean, post_logvar, prior_mean, prior_logvar])
    
    #sample z from either posterior during training
    z = inference.Sampler()([post_mean, post_logvar])
    
    #concat z and input (concat for each element in sequence)
        #broadcast z (BATCH_SZ, LATENT_DIM) to shape (BATCH_SZ, MAX_LEN_TARGET ,LATENT_DIM)
    z = layers.Reshape((1, LATENT_DIM))(z)
    broadcast_shape = tf.where([True, False, True], tf.shape(z), [0, MAX_LEN_TARGET, 0])
    broadcasted_z = layers.Lambda(broadcast_z)([z, broadcast_shape])
    dec_lstm_input = layers.Concatenate(axis = 2)([t_embedded, broadcasted_z])
    
    #set initial state for decoder
    initial_hidden_st = layers.Dense(HIDDEN_DIM*2, activation = 'tanh', name = 'get_dec_init_hidden')(back_h)
    initial_carry_st = layers.Dense(HIDDEN_DIM*2, activation = 'tanh', name = 'get_dec_init_carry')(back_c)
    
    #decoder LSTM
    dec_lstm = layers.LSTM(HIDDEN_DIM*2, return_sequences = True, name = 'dec_lstm')
    dec_lstm_output = dec_lstm(dec_lstm_input, initial_state = [initial_hidden_st, initial_carry_st])
    
    #attention mechanism
    #Dot axes param (2,2): dot axis 2 of first with axis 2 of second, equivalent to transposing the second one
    #(batch_size, MAX_LEN_TARGET, HIDDEN_DIM*2) dot (batch_size, MAX_LEN_SOURCE, HIDDEN_DIM*2)
    dot_products = layers.Dot(axes = (2,2))([dec_lstm_output, s_annotation_vectors])
    #take softmax of each row of matrix product
    alphas = layers.Softmax(axis = -1)(dot_products)
    #(batch_size, MAX_LEN_TARGET, MAX_LEN_SOURCE) dot (batch_size, MAX_LEN_SOURCE, HIDDEN_DIM*2)
    context_vectors = layers.Dot(axes = (2,1))([alphas, s_annotation_vectors]) #(batch_size, MAX_LEN_TARGET, HIDDEN_DIM*2)
    
    #concat with dec_lstm_output and z --> (batch_size, MAX_LEN_TARGET, HIDDEN_DIM*4 + LATENT_DIM)
    attention_pre_tanh = layers.Concatenate(axis = -1)([dec_lstm_output, context_vectors, broadcasted_z])
    attention_output = layers.Activation('tanh')(attention_pre_tanh)
    
    #get unnormalized p(yj | y<j, x, z) (softmax to be applied when getting actual translation)
    logits = layers.Dense(t_vocab_size)(attention_output)
    
    logits = loss.ReconstructionLossLayer()([target_sentence, logits])
    
    model = Model([source_sentence, target_sentence], [logits, post_mean, post_logvar, prior_mean, prior_logvar])
      
    return model                                                       


    #old decoding
    '''
    #decoder = decoding_example.Decoder(t_vocab_size, EMBED_DIM, HIDDEN_DIM*2, BATCH_SZ) ###decoder example directly from tutorial
    decoder = decoding.Decoder(t_vocab_size, HIDDEN_DIM*2, BATCH_SZ, target_embedding, 'decoder')
    decoder.attention_mechanism.setup_memory(s_annotation_vectors)
    
    #set initial state for decoder
    initial_hidden_st = layers.Dense(HIDDEN_DIM*2, activation = 'tanh', name = 'get_dec_init_hidden')(back_h)
    initial_carry_st = layers.Dense(HIDDEN_DIM*2, activation = 'tanh', name = 'get_dec_init_carry')(back_c)
    
    initial_state = decoder.build_initial_state(BATCH_SZ, [initial_hidden_st, initial_carry_st], tf.float32)

    output = decoder([t_embedded, z], initial_state)  
    #output = decoder(target_sentence, initial_state) ###decoder example directly from tutorial
    
    #translation is a list of word indices
    #translation shape = (batch_size, len_of_sentence)
    translation = output.sample_id
    #logits is output of final dense layer in decoder, unnormalized p(yj | y<j, x, z)
    #logits shape = (batch_size, len_of_sentence, vocab_size)
    logits = output.rnn_output
    '''                                                     


#take trained model and translate with it
def translate(sentences, trained_model, tokenizer_s, tokenizer_t):
    
    s_words = tokenizer_s.word_index
    t_words = tokenizer_t.word_index
    s_vocab_size = len(s_words) + 1 #add 1 because word index does not include padding
    t_vocab_size = len(t_words) + 1
    
    #get sequences from sentences
    sentences = dataset.get_sequences(sentences, tokenizer_s)
    batch_sz = sentences.shape[0]
    
    #word embeddings
    source_embedding = trained_model.get_layer('source_embedding')
    s_embedded = source_embedding(sentences)
    
    #encoding with bidirectional lstm
    source_encoder = trained_model.get_layer('source_encoder')
    s_annotation_vectors, _, _, back_h, back_c = source_encoder(s_embedded)
    
    #infererence
    infer_prior = trained_model.get_layer('infer_prior')
    prior_mean, prior_logvar = infer_prior(s_annotation_vectors)
    z = inference.Sampler()([prior_mean, prior_logvar])
    z = tf.reshape(z, [batch_sz, 1, LATENT_DIM]) #have to add dim for timstep
    
    dec_lstm =  trained_model.get_layer('dec_lstm')
    dec_lstm.return_state = True
    target_embedding = trained_model.get_layer('target_embedding')
    
    #hidden and carry states for decoder (manage for each step in decoding loop)
    hidden_st = layers.Dense(HIDDEN_DIM*2, activation = 'tanh', name = 'get_dec_init_hidden')(back_h)
    carry_st = layers.Dense(HIDDEN_DIM*2, activation = 'tanh', name = 'get_dec_init_carry')(back_c)
    
    #start with start token
    start_token = t_words['<start>']
    end_token = t_words['<end>']
    
    #create start with length of sentences
    dec_input = tf.fill([batch_sz,1], start_token) #add dim for timestep = 1
    dec_input = tf.cast(dec_input, tf.int64)
    output = dec_input
    
    #decoding 
    for i in range(MAX_LEN_TARGET):
        
        dec_input_embedded = target_embedding(dec_input)
        #dec_input_embedded = tf.reshape(dec_input_embedded, [batch_sz, 1, EMBED_DIM])
        dec_lstm_input = layers.Concatenate(axis = -1)([dec_input_embedded, z])
        #one decoding step
            #ideally I would build an LSTMCell from the LSTM layer but I don't know how to do that
        dec_lstm_output, hidden_st, carry_st = dec_lstm(dec_lstm_input, initial_state = [hidden_st, carry_st])
        
        #attention mechanism
        #(batch_size, 1, HIDDEN_DIM*2) dot (batch_size, MAX_LEN_SOURCE, HIDDEN_DIM*2)
        dot_products = layers.Dot(axes = (2,2))([dec_lstm_output, s_annotation_vectors])
        #take softmax of each row of matrix product
        alphas = layers.Softmax(axis = -1)(dot_products)
        #(batch_size, 1, MAX_LEN_SOURCE) dot (batch_size, MAX_LEN_SOURCE, HIDDEN_DIM*2)
        context_vectors = layers.Dot(axes = (2,1))([alphas, s_annotation_vectors]) #(batch_size, 1, HIDDEN_DIM*2)

        #concat with dec_lstm_output and z --> (batch_size, 1, HIDDEN_DIM*4 + LATENT_DIM)
        attention_pre_tanh = layers.Concatenate(axis = -1)([dec_lstm_output, context_vectors, z])
        attention_output = layers.Activation('tanh')(attention_pre_tanh)

        #get unnormalized p(yj | y<j, x, z) (softmax to be applied when getting actual translation)
        logits = layers.Dense(t_vocab_size)(attention_output)
        
        #apply softmax to normalize and get p(yj | y<j, x, z)
        word_probs = layers.Softmax(axis = -1)(logits) #(batch_size, 1, HIDDEN_DIM*4 + LATENT_DIM)
        
        #argmax to get next dec_input
        dec_input = tf.math.argmax(word_probs, -1)
                           
        output = tf.concat([output, dec_input],-1)
    
    #manipulate output so that last token is <end>
    output = output.numpy()
    end_tokens = np.full((batch_sz,1), end_token)
    output = np.append(output, end_tokens, -1)
    end_indices = np.argmax(output==end_token, -1)
    
    indices = np.arange(output.shape[1])
    output = output.tolist()
    for i in range(batch_sz):
        end_index = end_indices[i]
        output[i] = output[i][1:end_index]
    
    translation = tokenizer_t.sequences_to_texts(output)
    
    return translation, output

    #old decoding
    '''
    
    #build decoder with greedy sampler, and trained decoder parts
    greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler(target_embedding)    
    decoder = trained_model.get_layer('decoder')
    decoder_instance = decoding.VariationalBasicDecoder(cell=decoder.rnn_cell, sampler=greedy_sampler, output_layer=decoder.fc)
    decoder.attention_mechanism.setup_memory(s_annotation_vectors)
    
    #set initial state of new decoder
    initial_hidden_st = trained_model.get_layer('get_dec_init_hidden')(back_h)
    initial_carry_st = trained_model.get_layer('get_dec_init_carry')(back_c)
    initial_state = decoder.build_initial_state(batch_sz, [initial_hidden_st, initial_carry_st], tf.float32)
    
    #call decoder using start and end tokens
    start_tokens = tf.fill(batch_sz, t_words['<start>'])
    end_token = t_words['<end>']
    outputs, _, _ = decoder_instance([z,z], start_tokens=start_tokens, end_token=end_token, initial_state=initial_state) #should ignore inputs[0] because embedding passed to greedy embedding sampler, need it because functional API requires symbolic input
    translation = outputs.sample_id
    
    translator = Model(source_sentence, translation)
    return translator
    '''


#Lambdas

#https://stackoverflow.com/questions/57716363/explicit-broadcasting-of-variable-batch-size-tensor
#broadcast z (BATCH_SZ, LATENT_DIM) to shape (BATCH_SZ, MAX_LEN_TARGET ,LATENT_DIM)
def broadcast_z(args):
    z, broadcast_shape = args
    
    return tf.broadcast_to(z , broadcast_shape)
