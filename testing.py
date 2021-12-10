import dataset, model
from nltk.translate import bleu_score

dirname = 'jesc_prepared/test'
tokenizer_dir = 'tokenizers'

tokenizer_s, tokenizer_t = dataset.get_tokenizers(dirname, tokenizer_dir)
s_words = tokenizer_s.word_index
t_words = tokenizer_t.word_index
s_test, t_test = dataset.read_from_dir(dirname)

#in real evaluation, this line would be reading model from file
translator_model = model.build_translation_model(s_words, t_words)

translations, _ = model.translate(s_test, translator_model, tokenizer_s, tokenizer_t)

hypotheses = [translation.split() for translation in translations]
references = [[t_test_sent.split()] for t_test_sent in t_test]

corpus_bleu = bleu_score.corpus_bleu(references, hypotheses)
    
print(corpus_bleu)
    