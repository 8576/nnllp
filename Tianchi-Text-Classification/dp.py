# coding: UTF-8
import os
import torch
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import time
from datetime import timedelta
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader


MAX_VOCAB_SIZE = 10000
UNK, PAD = '<UNK>', '<PAD>'

class Createds(Dataset):
    def __init__(self, data_file, config) -> None:
        self.pad_size = config.pad_size
        self.n_gram_vocab = config.n_gram_vocab
        self.df = pd.read_csv(os.path.join("THUCNews", "data", data_file), sep="\t")
        self.tokenizer = lambda x: x.split(' ')
        self.device = torch.device("cuda")
        if os.path.exists(config.vocab_path):
            self.vocab = pickle.load(open(config.vocab_path, 'rb'))
        else:
            # self.vocab = self.build_vocab(config.train_path, tokenizer=self.tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
            raise ValueError("No such vocab {}\n Please create vocab file by build_vocab function".format(config.vocab_path))
        pickle.dump(self.vocab, open(config.vocab_path, 'wb'))
        print(f"Vocab size: {len(self.vocab)}")
    
    def __len__(self):
        return self.df.__len__()
    
    def _biGramHash(self, sequence, t, buckets):
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        return (t1 * 14918087) % buckets

    def _triGramHash(self, sequence, t, buckets):
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        t2 = sequence[t - 2] if t - 2 >= 0 else 0
        return (t2 * 14918087 * 18408749 + t1 * 14918087) % buckets

    def build_vocab(self, file_path, tokenizer, max_size, min_freq):
        vocab_dic = {}
        df = pd.read_csv(file_path, sep="\t")
        for _, line in tqdm(df.iterrows()):
            # print(line)
            content = line["text"]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
        return vocab_dic
    
    def __getitem__(self, index):
        x, y = self.df.iloc[index].text, self.df.iloc[index].label
        # x, y = self.df.iloc[index, 1], self.df.iloc[index, 0]
        # content, label = line["text"], line["label"]
        words_line = []
        token = self.tokenizer(x)
        seq_len = len(token)
        if len(token) < self.pad_size:
            token.extend([PAD] * (self.pad_size - seq_len))
        else:
            token = token[:self.pad_size]
            seq_len = self.pad_size
        # word to id
        for word in token:
            words_line.append(self.vocab.get(word, self.vocab.get(UNK)))

        # fasttext ngram
        buckets = self.n_gram_vocab
        bigram = []
        trigram = []
        # ------ngram------
        for i in range(self.pad_size):
            bigram.append(self._biGramHash(words_line, i, buckets))
            trigram.append(self._triGramHash(words_line, i, buckets))
        # -----------------
        # contents.append((words_line, int(label), seq_len, bigram, trigram))

        x = torch.LongTensor(words_line).to(self.device)
        # y = torch.LongTensor([y]).to(self.device)
        y = torch.tensor(y).to(self.device)
        bigram = torch.LongTensor(bigram).to(self.device)
        trigram = torch.LongTensor(trigram).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        # seq_len = torch.LongTensor([seq_len]).to(self.device)
        seq_len = torch.tensor(seq_len).to(self.device)
        # print("y.shape", y.shape)
        # print("x.sahpe", x.shape)
        # print("bigram.shape", bigram.shape)
        return (x, seq_len, bigram, trigram), y



def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    df = pd.read_csv(file_path, sep="\t")
    for _, line in tqdm(df.iterrows()):
        # print(line)
        content = line["text"]
        for word in tokenizer(content):
            vocab_dic[word] = vocab_dic.get(word, 0) + 1
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
    vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def build_dataset(config, ues_word):
    # if ues_word:
    tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    # else:
    # tokenizer = lambda x: [y for y in x]  # char-level
    if os.path.exists(config.vocab_path):
        vocab = pickle.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pickle.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}")

    def biGramHash(sequence, t, buckets):
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        return (t1 * 14918087) % buckets

    def triGramHash(sequence, t, buckets):
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        t2 = sequence[t - 2] if t - 2 >= 0 else 0
        return (t2 * 14918087 * 18408749 + t1 * 14918087) % buckets

    def load_dataset(path, pad_size=32):
        contents = []
        # with open(path, 'r', encoding='UTF-8') as f:
        df = pd.read_csv(path, sep="\t")
        df = shuffle(df)
        for _, line in tqdm(df.iterrows()):
            # lin = line.strip()
            # if not lin:
            #     continue
            # content, label = lin.split('\t')
            content, label = line["text"], line["label"]
            words_line = []
            token = tokenizer(content)
            seq_len = len(token)
            if pad_size:
                if len(token) < pad_size:
                    token.extend([PAD] * (pad_size - len(token)))
                else:
                    token = token[:pad_size]
                    seq_len = pad_size
            # word to id
            for word in token:
                words_line.append(vocab.get(word, vocab.get(UNK)))

            # fasttext ngram
            buckets = config.n_gram_vocab
            bigram = []
            trigram = []
            # ------ngram------
            for i in range(pad_size):
                bigram.append(biGramHash(words_line, i, buckets))
                trigram.append(triGramHash(words_line, i, buckets))
            # -----------------
            contents.append((words_line, int(label), seq_len, bigram, trigram))
        return contents  # [([...], 0), ([...], 1), ...]
    dataset = load_dataset(config.train_path, config.pad_size)
    # dev = load_dataset(config.dev_path, config.pad_size)
    # test = load_dataset(config.test_path, config.pad_size)
    datasize = dataset.__len__()
    trainsize = int(datasize * 0.7)
    devsize = int(datasize * 0.2)
    train = dataset[:trainsize]
    dev = dataset[trainsize:trainsize + devsize]
    test = dataset[trainsize + devsize:-1]
    

    return vocab, train, dev, test




class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数 
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        # xx = [xxx[2] for xxx in datas]
        # indexx = np.argsort(xx)[::-1]
        # datas = np.array(datas)[indexx]
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        bigram = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        trigram = torch.LongTensor([_[4] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len, bigram, trigram), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

if __name__ == "__main__":
    '''提取预训练词向量'''
    vocab_dir = "./THUCNews/data/vocab.pkl"
    pretrain_dir = "./THUCNews/data/sgns.sogou.char"
    emb_dim = 300
    filename_trimmed_dir = "./THUCNews/data/vocab.embedding.sougou"
    word_to_id = pkl.load(open(vocab_dir, 'rb'))
    embeddings = np.random.rand(len(word_to_id), emb_dim)
    f = open(pretrain_dir, "r", encoding='UTF-8')
    for i, line in enumerate(f.readlines()):
        # if i == 0:  # 若第一行是标题，则跳过
        #     continue
        lin = line.strip().split(" ")
        if lin[0] in word_to_id:
            idx = word_to_id[lin[0]]
            emb = [float(x) for x in lin[1:301]]
            embeddings[idx] = np.asarray(emb, dtype='float32')
    f.close()
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)
