import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

#jesc data processed on local machine before using this file in remote machine

#include filters argument to filter only certain punctuation and not all of them
        #https://stackoverflow.com/questions/49073673/include-punctuation-in-keras-tokenizer
        #japanese punctuation already taken care of (do not need to add spaces)
punc_excluded = '~"#$%&()*+-/=@[\\]^_`{|}~\t\n'
punc_tokenized = '.,:;!?'


#requirements:
    #file of source sentences 
    #file of target sentences 
#maybe add way to incorporate multiple references of target
def read_from_dir(dirname):
    datadir = dirname
    s_fname = os.path.join(datadir, 'source_sent.txt')
    s_f = open(s_fname)
    source_sentences = s_f.read().split('\n')
    s_f.close()
    t_fname = os.path.join(datadir, 'target_sent.txt')
    t_f = open(t_fname)
    target_sentences = t_f.read().split('\n')
    t_f.close()
    
    source_len = len(source_sentences)
    target_len = len(target_sentences)
    
    if source_len != target_len:       
        #truncate target_sequences to be same as source_sequences
        if target_len > source_len:
            target_sentences = target_sentences[:source_len]
        #truncate source_sequences to be same as target_sequences
        if source_len > target_len:
            source_sentences = source_sentences[:target_len]
    
    return source_sentences, target_sentences

#preprocess sentences
def preprocess_text(sentences):#, punc_tokenized):
    if type(sentences) is not list:
        sentences = [sentences]
    
    #add spaces to punctuation
    sentences = [re.sub(r'(['+punc_tokenized+'])', r' \1 ', sent) for sent in sentences]
    #take away all '<' '>' to avoid confusion with '<start>' and '<end>'
    sentences = [re.sub(r'([<>])', r'', sent) for sent in sentences] #experiment and check
    #insert '<start>' and '<end>' tokens
    sentences = ['<start> ' + sent + ' <end>' for sent in sentences]
    
    return sentences

#get tokenizers based on data in given dir
def get_tokenizers(data_dir, save_dir):
    
    #read object directly from save_dir if available
    #if directory exists, so much pickle files
    tok_s_fname = os.path.join(save_dir, 'tokenizer_s.pickle')
    tok_t_fname = os.path.join(save_dir, 'tokenizer_t.pickle')
    if os.path.isdir(save_dir):
        with open(tok_s_fname, 'rb') as f:
            tokenizer_s = pickle.load(f)
        with open(tok_t_fname, 'rb') as f:
            tokenizer_t = pickle.load(f)
        return tokenizer_s, tokenizer_t
    
    if data_dir is None:
        print("data_dir must be provided")
        return None
    print("Creating new Tokenizers")
    
    #read and preprocess
    source_sentences, target_sentences = read_from_dir(data_dir)
    source_sentences = preprocess_text(source_sentences)
    target_sentences = preprocess_text(target_sentences)
    
    #fit tokenizers
    tokenizer_s = Tokenizer(filters = punc_excluded, oov_token = '<unk>')
    tokenizer_t = Tokenizer(filters = punc_excluded, oov_token = '<unk>')
    tokenizer_s.fit_on_texts(source_sentences)
    tokenizer_t.fit_on_texts(target_sentences)
    
    if save_dir is None:
        print("save_dir must be provided")
        return None
    
    #write tokenizers to save_dir
    os.mkdir(save_dir)
    with open(tok_s_fname, 'wb') as f:
        pickle.dump(tokenizer_s, f, pickle.HIGHEST_PROTOCOL)
    with open(tok_t_fname, 'wb') as f:
        pickle.dump(tokenizer_t, f, pickle.HIGHEST_PROTOCOL)
    
    return tokenizer_s, tokenizer_t   
    
#create sequences data (split and with start and end tokens)
def get_sequences_from_dir(data_dir, tokenizer_s, tokenizer_t, seq_len=50):
    
    source_sentences, target_sentences = read_from_dir(data_dir)
    
    source_sequences = get_sequences(source_sentences, tokenizer_s)
    target_sequences = get_sequences(target_sentences, tokenizer_t)
    
    #print(tokenizer_s.sequences_to_texts(source_sequences))
    #print(tokenizer_t.sequences_to_texts(target_sequences))    
        
    return source_sequences, target_sequences
    

def get_sequences(sentences, tokenizer, seq_len=50):
    #preprocess
    sentences = preprocess_text(sentences)
    
    #run keras tokenizer on sentences
    sequences = tokenizer.texts_to_sequences(sentences)
    
    #pad the sequences to make them all the same length
    #is there an issue with the model and word index for padding?
    sequences = pad_sequences(sequences, maxlen = seq_len, padding = 'post')

    return sequences
        
#generator? 