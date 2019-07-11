from collections import Counter

from gensim.models import word2vec
from tqdm import tqdm

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def load_data(data_path):
    with open(data_path, 'r') as f:
        lines = f.readlines()
        data = [line.strip().split('_') for line in tqdm(lines)]
        return data


def token_count(sentences, token_count_file):
    counter = Counter([tok for sent in sentences for tok in sent])
    with open(token_count_file, 'w') as f:
        f.write('token,count\n')
        for item in tqdm(counter.items()):
            f.write(str(item[0])+','+str(item[1]))
            f.write('\n')


logging.info('Loading raw corpus...')
raw_sentences = load_data('../data/corpus.txt')

logging.info('Counting corpus tokens...')
token_count(raw_sentences, '../processed_data/corpus_token_count.txt')

logging.info('Word2vec model training...')
model = word2vec.Word2Vec(raw_sentences, min_count=3, workers=12)
model.wv.save_word2vec_format('../processed_data/w2v_v1.bin', binary=True)

logging.info('Testing similarity...')
print(model.similarity('2240', '7105'))
