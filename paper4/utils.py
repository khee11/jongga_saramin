import numpy as np
import time
import pickle
import os
import json
from nltk.tokenize import WordPunctTokenizer,TweetTokenizer,word_tokenize
from tqdm import tqdm
from collections import OrderedDict,Counter,defaultdict
from itertools import chain

def info_return(data_type):
    """
    data path, class and text column index, total size of data, number of classes
    """
    info_dict = {}
    
    if data_type == "dbpedia":
        info_dict["path"] = "data/dbpedia_csv"
        info_dict["skip_first"] = False
        info_dict["class_idx"] = 0
        info_dict["text_idx"] = 1
        info_dict["num_cols"] = 2
        info_dict["start_from"] = 1
        info_dict["num_class"] = 14
        info_dict["size"] = 560000 # 560,000
        
    elif data_type == "agnews":
        info_dict["path"] = "data/agnew_csv"
        info_dict["skip_first"] = True
        info_dict["class_idx"] = 0
        info_dict["text_idx"] = 1
        info_dict["num_cols"] = 2
        info_dict["start_from"] = 1
        info_dict["num_class"] = 4
        info_dict["size"] = 120000 # 120,000
        
    elif data_type == "yahoo":
        info_dict["path"] = "data/yahoo_answers_csv"
        info_dict["skip_first"] = False
        info_dict["class_idx"] = 0
        info_dict["text_idx"] = 1
        info_dict["num_cols"] = 2
        info_dict["start_from"] = 1
        info_dict["num_class"] = 10
        info_dict["size"] = 1400000 # 1,400,000

    elif data_type == "yelp":
        info_dict["path"] = "data/yelp_review_polarity_csv"
        info_dict["skip_first"] = False
        info_dict["class_idx"] = 0
        info_dict["text_idx"] = 1
        info_dict["num_cols"] = 2
        info_dict["start_from"] = 1
        info_dict["num_class"] = 2
        info_dict["size"] = 560000 # 560,000
        
    return info_dict

def return_tokenizer(token_type):
    if token_type=="word":
        return word_tokenize
        #tokenizer = WordPunctTokenizer()
        #return tokenizer.tokenize
    elif token_type=="tweet":
        tokenizer = TweetTokenizer()
        return tokenizer.tokenize

def remove_title(data_type):
    info = info_return(data_type)
    if info["skip_first"]:
        with open(info["path"]+"/train.csv") as f:
            train_csv = f.readlines()
        del train_csv[0]
        
        with open(info["path"]+"/train.csv","w") as f:
            for line in train_csv:
                f.write(line)             

        with open(info["path"]+"/test.csv") as f:
            train_csv = f.readlines()
        del train_csv[0]
        
        with open(info["path"]+"/test.csv","w") as f:
            for line in train_csv:
                f.write(line)             

# tokenizer에 의해, 새롭게 재구성된 corpus. tokenizer로 token화된 단어들이 ' '로 어이짐 (문장은 newline으로 뛰움)
def build_corpus_from_csv(data_type,tokenizer_type,is_lab_only=True,is_save=True):
    corpus = []
    info = info_return(data_type)
    tokenizer = return_tokenizer(tokenizer_type)
    out_path = f"pretrained/{data_type}/corpus.txt"
    
    if info["skip_first"]:
        print("*"*70+"\n"+"This dataset contains basically title (head) as its first line.\nBe sure to remove it. You can remove the line in preprocess.py\n"+"*"*70)
        time.sleep(3)
    
    print("Turn {} into its corpus".format(info["path"]+"/train_lab.csv"))
    with open(info["path"]+"/train_lab.csv","r") as f:
        for i,line in enumerate(tqdm(f)):
            splitted = line.lower().strip().split(",",1)
            text = splitted[info["text_idx"]].replace('"','').replace(r'\n','').strip(" ") 
            tokens = tokenizer(text)
            text = " ".join(tokens)
            corpus.append(text) # each sentence (or document)
    
    if not is_lab_only:
        print("Turn {} into its corpus".format(info["path"]+"/train_unlab.csv"))
        with open(info["path"]+"/train_unlab.csv","r") as f:
            for i,line in enumerate(tqdm(f)):
                splitted = line.lower().strip().split(",",1)
                text = splitted[info["text_idx"]].replace('"','').replace(r'\n','').strip(" ") 
                tokens = tokenizer(text)
                text = " ".join(tokens) # Again into sentence
                corpus.append(text) # each sentence (or document)
    
    # useful if train glove with our dataset
    if is_save:
        with open(out_path,"w") as f:
            for doc in corpus:
                f.write(doc+" \n")
    return corpus

def save_vocab_from_corpus(corpus,out_path,min_freq=1):
    """
    corpus : list of lists, each of which is a sentence
    out_path : save path of vocab, txt format
    min_freq : only tokens whose count greater than `min_freq` are saved in the vocab
    """
    #all_tokens = []
    count = Counter()
    if type(corpus) == str:
        corpus = open(corpus)
    elif type(corpus) == list :
        corpus = corpus
        
    for line in corpus:
        line = line.strip().split()
        #all_tokens.extend(line) # 이게 잘 작동하는지 체크
        count.update(line)
    count = count.most_common()#Counter(all_tokens).most_common()
    
    with open(out_path,"w") as f:
        for word,cnt in count:
            if cnt > min_freq:
                f.write(word+" "+str(cnt)+"\n")

# Loca vocabulary from text file, including speical tokesn and give number
def load_vocab_from_txt(path,is_ordered=False):
    # path : argparse로
    vocab = {
        "<pad>":0,
        "<unk>":1,
        "<eos>":2
    }
    count_prev = int(1e10)
    with open(path) as f:
        for line in f:
            word,count = line.strip().split()
            count = int(count)
            vocab[word] = len(vocab)
            if is_ordered:
                assert count_prev>=count,"vocabulary text file is not aligned as in the count-decreasing order"
            else:
                pass
            count_prev = count
    return vocab

# 데이터로 glove를 직접 훈련시 생기는 vectors을 활용하는 함수
def embedding_matrix_from_txt(in_path,out_path,vocab,dim,is_return=False):
    # dim,in_path and out_path는 argparse로
    vocab_size = len(vocab)
    vectors = np.zeros(shape=[vocab_size,dim],dtype=np.float32)
    with open(in_path) as f:
        for line in tqdm(f):
            splitted = line.split()
            word = splitted[0]
            try:
                word_index = vocab.get(word)
                vector = list(map(lambda x : float(x),splitted[1:]))
                vectors[word_index] = vector
            except:
                print("Word {}, in vocabulary, but not in an embedding file")
                continue
    # Initialize unknown embeddings
    vectors[:3] = np.random.uniform(-.01,.01,[3,dim])
    if out_path is None:
        pass
    else:
        np.save(out_path,vectors)
    if is_return:
        return vectors

# wrapper func for different format of vectors file
def load_embedding_matrix(in_path,vocab):
    extension = in_path.split(".")[-1]
    if extension == "npy":
        vectors = np.load(in_path)
    elif extension == "txt":
        vectors = embedding_matrix_from_txt(in_path,None,vocab,300,True)    
    return vectors

# 주어진 vocab에 해당하는 embedding matrix을 txt file로부터 load해줌
def match_vectors_with_vocab_from_txt(txt_path,vocab,dim,save_path=None):
    """
    txt_path is one in 
    Vocab in the parameters should be include special tokens like, <pad>,<unk>,<eos>.
    """
    vectors = np.zeros([len(vocab),dim],dtype=np.float32)
    matched_index = []
    with open(txt_path) as f:
        for i,line in enumerate(tqdm(f)):
            splitted = line.strip().split(" ")
            word = splitted[0]
            vector = splitted[1:]
            try:
                index = vocab[word]                
            except:
                continue
            vectors[index] = list(map(lambda x:float(x),vector))
            matched_index.append(index)
    
    total = set(vocab.values())
    unmatched_index = list(total.difference(matched_index))
    # Initializae unmatched vectors with random values
    vectors[unmatched_index] = np.random.uniform(-0.01,0.01,[len(unmatched_index),dim])
    print("{} (out of {}) are unmatched".format(len(unmatched_index),len(total)))
    
    if save_path is not None:
        np.save(save_path,vectors)
    
    return vectors

def split_whole_data(data_type,num_labeled,num_unlabeled,num_val,seed):
    
    def per_label_text(path):
        return_dict = defaultdict(list)
        f = open(path)
        for line in f:
            split = line.split(",",1)
            label = int(split[0].strip('"'))
            text = split[1]#.replace('"','')
            return_dict[label].append(text)
        f.close()
        return return_dict
    # Get dictionary whose keys are label, value is a list of corresponding text
    info = info_return(data_type)
    return_dict = per_label_text(info["path"]+"/train.csv")
    if type(num_labeled)==float:
        num_labeled = info["size"]*num_labeled
    
    assert info["num_class"] == len(return_dict.keys()),"# of classes differ"
    
    # Shuffle given a seed
    rng = np.random.RandomState(seed)
    for key,content in return_dict.items():
        rng.shuffle(content)
    
    f = open(info["path"]+"/train_lab.csv","w")
    f2 = open(info["path"]+"/eval.csv","w")
    f3 = open(info["path"]+"/train_unlab.csv","w")
    for label,content in return_dict.items():
        for i,text in enumerate(content):
            wrtie_content = str(label)+","+text
            if i < num_labeled:
                f.write(wrtie_content)
            elif i >= num_labeled and i< (num_labeled+num_val) : 
                f2.write(wrtie_content)
            elif i < (num_labeled+num_val+num_unlabeled):
                f3.write(wrtie_content)
    f.close()
    f2.close()
    f3.close()

def load_json(path,target):
    with open(path,"r") as f:
        obj = json.load(f)
    return obj[target]
