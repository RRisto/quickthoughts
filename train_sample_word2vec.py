from gensim.models import Word2Vec, KeyedVectors
from pathlib import Path

sentences = Path('data/cleaned.txt').read_text().split('\n')

sentences_tokenized = [sent.split() for sent in sentences]

model = Word2Vec(window=2, size=300)
model.build_vocab(sentences_tokenized)
model.train(sentences, total_examples=model.corpus_count, epochs=30, report_delay=1)

model.save('embedding_models/sample_w2v.bin')
word_vectors = model.wv
word_vectors.save('embedding_models/sample_w2v.kv')
