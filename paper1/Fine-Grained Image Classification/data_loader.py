# https://stackoverflow.com/questions/52099863/how-to-read-decode-tfrecords-with-tf-data-api
# https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models

import zipfile
from PIL import Image
import os,sys,pickle,re
from collections import defaultdict
import tensorflow as tf
import numpy as np
from konlpy.tag import Okt

"""
Goal is to build TFRecordFiles, and define loader function(return dataset or iterator)

0. Build numpy array for Embedding
1. Build a dictionary mapping image fname to something
    For images, fname -> raw_string
    For text, fname -> (padded) corresponding ids
    For label, fname -> class label(int, from 0,...,101) : total 102
2. Build TFRecord files
    description is for fname,label,images(raw_string), (padded) text
    You can build example proto per each fname
    Then filewriter would handle this
3. Define read(parsing)
    After building record files
    Define parser, and need preprocessing before putting into a model
    For example, normalize model(or resize), one_hot_vector for label
4. Build Dataset API
    dataset, and subsequent iterator
    
"""

# Define some directories or values 
# Chnage this with your need

IMG_ZIP = "jpg.zip"
IMG_DIR = os.path.join(os.getcwd(),"image_data")
TEXT_FILE = "flower_text_tagged.txt"
embd_file = "skigram1.txt"
embd_vocab_file="flower_dictionary2"
embd_np_file="pretrained_skipgram.npy"

TF_DIR = os.path.join(os.getcwd(),"tfrecord_data")

MAX_LEN = 50 # MAX Tokens length per each description
BATCH_SIZE=64
NUM_CLASS=102 # NOTE that class label starts from 0 to 101, total 102 classes

stop_pos = ["Josa","Conjunction","Determiner","Suffix","Alpha",'Punctuation']
stop_words = ["이","그","이다","이고","있는","은","는","있다","있습니다","있는데","의","과","꽃","가지","고","식물","하다","합니다","가지다","가지고","가지"]


# Define function for tfrecords
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def ids_to_sent(ids,vocab):
        x=list(map(lambda x:vocab[x],ids))
        return " ".join(x)

# the following functions are to be run on main() func
def build_embedding_mat(is_save=True):
    ### Build embedding matrix for text ###
    f=open(embd_file,'r')
    token_to_embd = {}
    for i,line in enumerate(f):
        if i==0:
            VOCAB_SIZE,EMBD_DIM=line.strip().split()
            VOCAB_SIZE,EMBD_DIM=int(VOCAB_SIZE),int(EMBD_DIM)
            continue
    line=line.split()
    token_to_embd[line[0]] = \
        list(map(lambda x: float(x),line[1:]))
    f.close()

    f= open("flower_dictionary2","rb")
    vocab = pickle.load(f)
    vocab_inv= dict(vocab)
    vocab = {val:key for key,val in vocab_inv.items()} 
    f.close()

    assert VOCAB_SIZE==len(vocab),"Vocabulary size should be same"

    embd_mat=np.zeros(shape=[VOCAB_SIZE,EMBD_DIM],dtype=np.float32)
    keys2=list(vocab.keys())
    for token in token_to_embd.keys():
        assert token in keys2, f"{token} not in vocabulary"
        idx=vocab[token]
        embd_mat[idx]=token_to_embd[token]

    if is_save:
        np.save(embd_np_file,embd_mat)
    
    return vocab,vocab_inv
   

def img_to_dict():
    ### name2img_raw ###
    if os.path.exists(IMG_DIR):
        print("Images are already unzipped")
    else:
        print("Unzip the Image files")
        img_zip = zipfile.ZipFile(IMG_ZIP)
        img_zip.extractall(IMG_DIR)
        img_zip.close()

    img_filenames=os.listdir(IMG_DIR)

    def image_to_raw(fdir,size=[224,224]):
        image=Image.open(fdir)
        image_raw=image.resize(size).tobytes()
        image.close()
    
        return image_raw
    
    img_dict = dict()
    for fname in img_filenames:
        img_dict[fname[:-4]]=image_to_raw(os.path.join(IMG_DIR,fname))

    return img_dict

def txt_to_dict(vocab,vocab_inv):
    """
    Use vocab and vocab_inv from build_embedding_mat() for integrity purpose
    """
    ### name2text_id_padded ###
    f= open(TEXT_FILE,"rb")
    fname_to_txt = defaultdict(str)
    
    for line in f:
        _txt=line.decode("utf-8-sig").strip()
        fname=_txt[:11]
        string = _txt[12:].replace("\ufeff","")
        fname_to_txt[fname]+=string+" "
    f.close()

    # Now turn a string into a list of corresponding ids

    def tokenize_and_id(txt,_vocab,tokenizer,max_len):
        txt_re = re.sub(r"[^ㄱ-힣0-9]"," ",txt)
        tokens=tokenizer(txt_re)
        tokens=[token for token,pos in tokens \
            if pos not in stop_pos and token not in stop_words]
        tokens=list(map(lambda x:_vocab.get(x,vocab["과"]),tokens))
        tokens=list(filter(lambda x:x!=vocab["과"],tokens))
        return tokens[:max_len]+[vocab["은"]]*(max_len-len(tokens))

    word_tokenize=Okt().pos
    ids_dict={key:tokenize_and_id(val,vocab,word_tokenize,MAX_LEN) \
        for key,val in fname_to_txt.items()}
    
    print("An example : \n",ids_to_sent(ids_dict["image_00001"],vocab_inv))
    return ids_dict

def label_to_dict():
    ### name2labels ###
    f= open("flower_class.txt","r")
    label_dict=dict()
    _compile=re.compile(r"\d+")
    for line in f:
        matobj=_compile.search(line[:11])
        label = int(matobj.group(0))
        fname=line[12:].strip()
        label_dict[fname]=label-1 # For one hot vector
    f.close()
    
    return label_dict

### Turn them into TFRecord files ###
def tfrecord_writer(img_dict,ids_dict,label_dict):

    def fname_to_list(fdir):
        lst=[]
        with open(fdir,"r") as f:
            for line in f:
                lst.append(line[:-5])        
        return lst
    
    train_list=fname_to_list("train_image.txt")
    eval_list=fname_to_list("val_image.txt")
    test_list=fname_to_list("test_image.txt")
    
    def example_func(img,ids,label,fname):
        example = tf.train.Example(features=tf.train.Features(feature={
            "img_raw":_bytes_feature(img),
            "text_ids":_int64_feature(ids),
            "label":_int64_feature([label]),
            "fname":_bytes_feature(bytes(fname,"utf-8"))
        }))
        return example
    
    tf_train_writer = tf.io.TFRecordWriter(TF_DIR+"/tfrecord_train")
    tf_eval_writer = tf.io.TFRecordWriter(TF_DIR+"/tfrecord_eval")
    tf_test_writer = tf.io.TFRecordWriter(TF_DIR+"/tfrecord_test")
    for idx,fname in enumerate(label_dict.keys()):
        example=example_func(img_dict[fname],ids_dict[fname],label_dict[fname],fname)
        assert not (fname in train_list and fname in eval_list),\
            f"A file({fname}) should be a train,eval or test" 
        if fname in train_list:
            tf_train_writer.write(example.SerializeToString())
        elif fname in eval_list:
            tf_eval_writer.write(example.SerializeToString())
        elif fname in test_list:
            tf_test_writer.write(example.SerializeToString())

    tf_train_writer.close()
    tf_eval_writer.close()
    tf_test_writer.close()
    print("TFRecord has been built, split into train,evaluation and test")


### Read and parse, preprocessing suitable for model ###
def tfrecord_loader_demo(vocab_inv):
    description={
        "img_raw":tf.io.FixedLenFeature([],tf.string),
        "text_ids":tf.io.FixedLenFeature([MAX_LEN],tf.int64),
        "label":tf.io.FixedLenFeature([],tf.int64),
        "fname":tf.io.FixedLenFeature([],tf.string)
    }

    def _parse_single_example(example_proto):
        parsed_feature=tf.io.parse_single_example(example_proto,description)
        img_raw = parsed_feature["img_raw"]
        text_ids = parsed_feature["text_ids"]
        label = parsed_feature["label"]
        fname = parsed_feature["fname"]
        
        image = tf.decode_raw(img_raw,tf.uint8)
        image = tf.reshape(image,[224,224,3])
        
        label = tf.one_hot(label,depth=NUM_CLASS)
        
        return image,text_ids,label,fname

    #dataset = tf.data.Dataset.TFRecordDataset(os.listdir(TF_DIR))
    dataset =  tf.data.TFRecordDataset(os.path.join(TF_DIR,"tfrecord_train"))
    dataset = dataset.map(_parse_single_example)
    dataset = dataset.repeat(1).shuffle(10000).batch(BATCH_SIZE)
    dataset = dataset.make_one_shot_iterator()
    image,text_ids,label,fname = dataset.get_next()


    with tf.Session() as sess:
        a,b,c,d=sess.run([image,text_ids,label,fname])

    print("A demo from tfrecord files:\n")
    filename=d[0].decode("utf-8")
    print(f"Filename : {filename}")
    Image.fromarray(a[0]).show()
    print(ids_to_sent(b[0],vocab_inv))
    print(c[0])
    print("Class :",label_dict[filename])

def data_loader(record_name):
    description={
        "img_raw":tf.io.FixedLenFeature([],tf.string),
        "text_ids":tf.io.FixedLenFeature([MAX_LEN],tf.int64),
        "label":tf.io.FixedLenFeature([],tf.int64),
        "fname":tf.io.FixedLenFeature([],tf.string)
    }
    
    def _parse_single_example(example_proto):
        parsed_feature=tf.io.parse_single_example(example_proto,description)
        img_raw = parsed_feature["img_raw"]
        text_ids = parsed_feature["text_ids"]
        label = parsed_feature["label"]
        fname = parsed_feature["fname"]
        
        image = tf.decode_raw(img_raw,tf.uint8)
        image = tf.reshape(image,[224,224,3])
        
        label = tf.one_hot(label,depth=NUM_CLASS)
        
        return image,text_ids,label,fname
    
    dataset =  tf.data.TFRecordDataset(os.path.join(TF_DIR,f"{record_name}"))
    dataset = dataset.map(_parse_single_example)
    return dataset
    #dataset = dataset.repeat(repeat).shuffle(10000).batch(BATCH_SIZE)
    #dataset = dataset.make_one_shot_iterator()
    #image,text_ids,label,fname = dataset.get_next()
    
    #return image,text_ids,label,fname

if __name__=="__main__":
    if not os.path.exists(embd_np_file):
        print("Saving a word embedding as numpy file...")
        vocab,vocab_inv=build_embedding_mat(is_save=True)
    else:
        print("Embedding has been already saved in this directory")
        vocab,vocab_inv=build_embedding_mat(is_save=False)
    
    if os.path.exists(TF_DIR):
        print(f"TFRecord files are saved in {TF_DIR}\n\rNo Need to create another ones")
        label_dict=label_to_dict()
    else:
        print("Serializing images...")
        img_dict=img_to_dict()
        print("Turn sentences into id tokens")
        ids_dict=txt_to_dict(vocab,vocab_inv)
        label_dict=label_to_dict()
        os.mkdir(TF_DIR)
        tfrecord_writer(img_dict,ids_dict,label_dict)

    print("Run tfrecord reader demo...\n")
    tfrecord_loader_demo(vocab_inv)
    

