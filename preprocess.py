from transformers import BertTokenizer
import os

def make_id_file(task, tokenizer):
    def make_data_strings(file_name):
        data_strings = []
        with open(os.path.join(file_name), 'r', encoding='utf-8') as f:
            id_file_data = [tokenizer.encode(line.lower()) for line in f.readlines()]
        for item in id_file_data:
            data_strings.append(' '.join([str(k) for k in item]))
        return data_strings
    
    print('it will take some times...')
    train_pos = make_data_strings('./data/sentiment.train.1')
    train_neg = make_data_strings('./data/sentiment.train.0')
    dev_pos = make_data_strings('./data/sentiment.dev.1')
    dev_neg = make_data_strings('./data/sentiment.dev.0')

    print('make id file finished!')
    return train_pos, train_neg, dev_pos, dev_neg

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_pos, train_neg, dev_pos, dev_neg = make_id_file('yelp', tokenizer)

import pandas as pd

test_df = pd.read_csv('./data/test_no_label.csv')
test_dataset = test_df['Id']

def make_id_file_test(tokenizer, test_dataset):
    data_strings = []
    id_file_data = [tokenizer.encode(sent.lower()) for sent in test_dataset]
    for item in id_file_data:
        data_strings.append(' '.join([str(k) for k in item]))
    return data_strings

test = make_id_file_test(tokenizer, test_dataset)

import pickle
open_file = open(os.path.join("./data/train_pos.pkl"), "wb")
pickle.dump(train_pos, open_file)
open_file.close()

open_file = open(os.path.join("./data/train_neg.pkl"), "wb")
pickle.dump(train_neg, open_file)
open_file.close()

open_file = open(os.path.join("./data/dev_pos.pkl"), "wb")
pickle.dump(dev_pos, open_file)
open_file.close()

open_file = open(os.path.join("./data/dev_neg.pkl"), "wb")
pickle.dump(dev_neg, open_file)
open_file.close()

open_file = open(os.path.join("./data/test.pkl"), "wb")
pickle.dump(test, open_file)
open_file.close()