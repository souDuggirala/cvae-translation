import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow_addons.seq2seq import AttentionWrapperState, AttentionWrapper, BasicDecoder, BasicDecoderOutput
#using tensorflow addons
#https://www.tensorflow.org/addons/api_docs/python/tfa/seq2seq
#https://www.tensorflow.org/addons/tutorials/networks_seq2seq_nmt

#https://www.analyticsvidhya.com/blog/2020/08/a-simple-introduction-to-sequence-to-sequence-models/

#how does this code guarentee that it stop predicting after <end>?

LATENT_DIM = 32
MAX_LEN_SOURCE = 50
MAX_LEN_TARGET = 50

class Decoder(tf.keras.Model):
    
    def __init__(self, vocab_size, dec_units, batch_sz, embedding_layer, name, attention_type='bahdanau'):
        
        super(Decoder, self).__init__(name = name)
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.attention_type = attention_type

        # Embedding Layer (reusing from model code
        self.embedding = embedding_layer

        #Final Dense layer on which softmax will be applied
        self.fc = tf.keras.layers.Dense(vocab_size)

        # Define the fundamental cell for decoder recurrent structure
        self.decoder_rnn_cell = tf.keras.layers.LSTMCell(self.dec_units)

        # Sampler
        #self.sampler = tfa.seq2seq.sampler.SampleEmbeddingSampler(embedding_fn = self.embedding) #GreedyEmbeddingSampler will do argmax instead
        self.sampler = tfa.seq2seq.sampler.TrainingSampler() #just chooses the next word in target sentence
        
        # Create attention mechanism with memory = None, memory set up by model.py for each input
        self.attention_mechanism = self.build_attention_mechanism(self.dec_units, None, self.batch_sz*[MAX_LEN_SOURCE], self.attention_type)

        # Wrap attention mechanism with the fundamental rnn cell of decoder
        self.rnn_cell = self.build_rnn_cell()

        # Define the decoder with respect to fundamental rnn cell
        self.decoder = VariationalBasicDecoder(self.rnn_cell, sampler=self.sampler, output_layer=self.fc)

    #why do you need batch_sz for this?
    def build_rnn_cell(self):
        rnn_cell = VariationalAttentionWrapper(self.decoder_rnn_cell, self.attention_mechanism, attention_layer_size = self.dec_units*2 + LATENT_DIM)
        return rnn_cell

    #memory comes from attention_mechanism.setup_memory(sample_output) in model.py
    #see if I even need the memory sequence length 
    def build_attention_mechanism(self, dec_units, memory, memory_sequence_length, attention_type='bahdanau'):
        return tfa.seq2seq.BahdanauAttention(units=dec_units, memory=memory)#, memory_sequence_length=memory_sequence_length)

    #get AttentionWrapperState version of encoder_state
    def build_initial_state(self, batch_sz, encoder_state, Dtype):
        decoder_initial_state = self.rnn_cell.get_initial_state(batch_size=batch_sz, dtype=Dtype) #returns initial zero tuple
        decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
        return decoder_initial_state

    #inputs is a list [target_sentence, z]
    def call(self, inputs, initial_state):
        #print("input[0] shape: " + str(inputs[0].shape))
        #print("expected shape: (batch_sz, sentence_length)\n")
        #inputs[0] = self.embedding(inputs[0])
        print("embedded input[0] shape: " + str(inputs[0].shape))
        print("expected shape: (batch_sz, sentence_length, embedding)\n")
        outputs, _, _ = self.decoder(inputs, initial_state=initial_state, sequence_length=self.batch_sz*[MAX_LEN_TARGET-1]) 
        return outputs
    
'''
Replaces cell_input_fn() from AttentionWrapper
Instead of combining last attention output and input, combines z and input
'''
def cell_input_fn_z(inputs, z):
    return tf.concat([inputs, z], -1)

'''
Copied from source method _compute_attention and modified to incorporate extra parameter z
'''
def attention_fn_z(
    attention_mechanism, cell_output, attention_state, attention_layer, z
):
    """Computes the attention and alignments for a given
    attention_mechanism."""
    alignments, next_attention_state = attention_mechanism(
        [cell_output, attention_state]
    )

    # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
    expanded_alignments = tf.expand_dims(alignments, 1)
    # Context is the inner product of alignments and values along the
    # memory time dimension.
    # alignments shape is
    #   [batch_size, 1, memory_time]
    # attention_mechanism.values shape is
    #   [batch_size, memory_time, memory_size]
    # the batched matmul is over memory_time, so the output shape is
    #   [batch_size, 1, memory_size].
    # we then squeeze out the singleton dim.
    context_ = tf.matmul(expanded_alignments, attention_mechanism.values)
    context_ = tf.squeeze(context_, [1])

    #CHANGED TO CONCAT Z AND APPLY TANH
    #TOOK AWAY ATTENTION_LAYER - to match paper code more
    if attention_layer is not None:
        attention = tf.math.tanh(tf.concat([cell_output, context_, z], -1)) #took away preset dense attention_layer in order to
    else:
        attention = context_

    return attention, alignments, next_attention_state
    
    
class VariationalAttentionWrapper(tfa.seq2seq.AttentionWrapper):
    
    def __init__(self, cell, attention_mechanism, attention_layer_size): 
        super().__init__(cell = cell, attention_mechanism = attention_mechanism, attention_layer_size = attention_layer_size)
        
        #I decided to define new functions to replace self._cell_input_fn and self._attention_fn in call(), because of the differing parameters
        self._cell_input_fn_z = cell_input_fn_z
        self._attention_fn_z = attention_fn_z
        
    '''    
    original source code slightly modified to use my own methods that use z
    copied the whole call method because I needed to change the parameters of the original self._cell_input_fn(inputs, attention) and self._attention_fn(attention_mechanism, cell_output, attention_state, attention_layer)
    '''
    def call(self, inputs, state, **kwargs): 
        
        if not isinstance(state, AttentionWrapperState):
            try:
                state = AttentionWrapperState(*state)
            except TypeError:
                raise TypeError(
                    "Expected state to be instance of AttentionWrapperState or "
                    "values that can construct AttentionWrapperState. "
                    "Received type %s instead." % type(state)
                )

        #CHANGE : each inputs is a tuple of tensors, for the actual input and then z
        actual_input = inputs[0]
        z = inputs[1]
        
        cell_inputs = self._cell_input_fn_z(actual_input, z) #REPLACED METHOD
        cell_state = state.cell_state
        cell_output, next_cell_state = self._cell(cell_inputs, cell_state, **kwargs)
        next_cell_state = tf.nest.pack_sequence_as(
            cell_state, tf.nest.flatten(next_cell_state)
        )

        cell_batch_size = cell_output.shape[0] or tf.shape(cell_output)[0]
        error_message = (
            "When applying AttentionWrapper %s: " % self.name
            + "Non-matching batch sizes between the memory "
            "(encoder output) and the query (decoder output).  Are you using "
            "the BeamSearchDecoder?  You may need to tile your memory input "
            "via the tfa.seq2seq.tile_batch function with argument "
            "multiple=beam_width."
        )
        with tf.control_dependencies(
            self._batch_size_checks(cell_batch_size, error_message)
        ):  # pylint: disable=bad-continuation
            cell_output = tf.identity(cell_output, name="checked_cell_output")

        if self._is_multi:
            previous_attention_state = state.attention_state
            previous_alignment_history = state.alignment_history
        else:
            previous_attention_state = [state.attention_state]
            previous_alignment_history = [state.alignment_history]

        all_alignments = []
        all_attentions = []
        all_attention_states = []
        maybe_all_histories = []
        for i, attention_mechanism in enumerate(self._attention_mechanisms):
            attention, alignments, next_attention_state = self._attention_fn_z(
                attention_mechanism,
                cell_output,
                previous_attention_state[i],
                self._attention_layers[i] if self._attention_layers else None,
                z
            ) #REPLACED METHOD
            alignment_history = (
                previous_alignment_history[i].write(
                    previous_alignment_history[i].size(), alignments
                )
                if self._alignment_history
                else ()
            )

            all_attention_states.append(next_attention_state)
            all_alignments.append(alignments)
            all_attentions.append(attention)
            maybe_all_histories.append(alignment_history)

        attention = tf.concat(all_attentions, 1)
        next_state = AttentionWrapperState(
            cell_state=next_cell_state,
            attention=attention,
            attention_state=self._item_or_tuple(all_attention_states),
            alignments=self._item_or_tuple(all_alignments),
            alignment_history=self._item_or_tuple(maybe_all_histories),
        )

        if self._output_attention:
            return attention, next_state
        else:
            return cell_output, next_state

'''subclassing to incorporate z'''
class VariationalBasicDecoder(tfa.seq2seq.BasicDecoder):
    
    def initialize(self, inputs, initial_state=None, **kwargs):
        
        """Initialize the decoder."""
        #print("Initial state at very beginning of initialize(): " + str(initial_state))
        # Assume the dtype of the cell is the output_size structure
        # containing the input_state's first component's dtype.
        self._cell_dtype = tf.nest.flatten(initial_state)[0].dtype
        initial_finished, initial_inputs, initial_state = self.sampler.initialize(inputs[0], **kwargs) + (initial_state,)#INDLUDED INDEX 0
        #print("Initializing initial inputs. input shape: " + str(initial_inputs.shape)) #+ " state: " + str(initial_state))
        #print("Old input shape: " + str(inputs[0].shape))
        initial_inputs = [initial_inputs, inputs[1]] #BRING Z TO INITIAL INPUTS
        return initial_finished, initial_inputs, initial_state #INDLUDED INDEX 0
    
    def step(self, time, inputs, state, training=None):
        
        #print(state)
        #print()
        
        print("Shape of inputs into cell: " + str(inputs[0].shape))
        print("expected: (batch_sz, word_embedding) = (3, 300) - single word\n")
        cell_outputs, cell_state = self.cell(inputs, state, training=training)
        #print("Shape of outputs from cell: " + str(cell_outputs.shape))
        #print("expected: (batch_sz, DEC_DIM*2 + LATENT) = (3, 1232) - hidden state/output;context vector;z\n")
        cell_state = tf.nest.pack_sequence_as(state, tf.nest.flatten(cell_state))
        #print("state from cell " + str(cell_state.cell_state))
        #print()
        
        if self.output_layer is not None:
            cell_outputs = self.output_layer(cell_outputs)
        #print("Shape of outputs after passing through dense: " + str(cell_outputs.shape))
        #print("expected: (batch_sz, VOCAB_SIZE) = (3, ?)\n")
        sample_ids = self.sampler.sample(
            time=time, outputs=cell_outputs, state=cell_state
        )
        (finished, next_inputs, next_state) = self.sampler.next_inputs(
            time=time, outputs=cell_outputs, state=cell_state, sample_ids=sample_ids
        )
        #print("Shape of next inputs from sampler: " + str(next_inputs.shape))
        #print("expected: (batch_sz, word_embedding) = (3, 300) - single word\n")
        #print("next state from sampler: " + str(next_state.cell_state))
        #print()
        
        outputs = BasicDecoderOutput(cell_outputs, sample_ids)
        next_inputs = [next_inputs,inputs[1]] #NEW LINE PASSING ALONG z TO NEXT STEP
        
        #print(next_state)
        
        return (outputs, next_state, next_inputs, finished)