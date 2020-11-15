import tensorflow as tf
import numpy as np
from utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data",type=str)
parser.add_argument("--tokenizer",type=str)
args= parser.parse_args()

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def write_tfrecord(in_path,out_path,data_type,vocab_name,tokenizer_type):
    info = info_return(data_type)
    vocab_path = f"pretrained/{data_type}/{vocab_name}.txt"
    vocab = load_vocab_from_txt(vocab_path)
    tokenizer = return_tokenizer(tokenizer_type)
    f = open(in_path,"r")
    tfwriter = tf.io.TFRecordWriter(out_path)   
    
    def example_func(tokens,label):
        example = tf.train.Example(
            features=tf.train.Features(
                feature={"tokens":tokens,"label":label}
            )
        )
        return example
        
    for line in tqdm(f):
        label,text= line.split(",",1)
        label = int(label.strip('"'))-1
        text = text.lower().replace('"','').strip()
        tokens = tokenizer(text)
        tokens = [vocab.get(token,1) for token in tokens]+[2]
        tokens =  _int64_feature(tokens)
        label =  _int64_feature([label])
        example = example_func(tokens,label)
        tfwriter.write(example.SerializeToString())
    f.close()

# For labeled only
print("Writing labeled dataset..")
info = info_return(args.data)
in_path = info["path"]+"/train_lab.csv"
out_path = info["path"]+"/train_lab_lab.tfrecord"
write_tfrecord(in_path,out_path,args.data,"vocab_lab",args.tokenizer)

#print("Writing unlabeled dataset...")
#in_path = info["path"]+"/train_unlab.csv"
#out_path = info["path"]+"/train_lab_unlab.tfrecord"
#write_tfrecord(in_path,out_path,args.data,"vocab_lab",args.tokenizer)

print("Writing evaluation dataset...")
in_path = info["path"]+"/eval.csv"
out_path = info["path"]+"/eval_lab.tfrecord"
write_tfrecord(in_path,out_path,args.data,"vocab_lab",args.tokenizer)

print("Writing Test dataset...")
in_path = info["path"]+"/test.csv"
out_path = info["path"]+"/test_lab.tfrecord"
write_tfrecord(in_path,out_path,args.data,"vocab_lab",args.tokenizer)

# For labeled + unlabeled
print("Writing labeled dataset..")
info = info_return(args.data)
in_path = info["path"]+"/train_lab.csv"
out_path = info["path"]+"/train_all_lab.tfrecord"
write_tfrecord(in_path,out_path,args.data,"vocab_all",args.tokenizer)

print("Writing unlabeled dataset...")
in_path = info["path"]+"/train_unlab.csv"
out_path = info["path"]+"/train_all_unlab.tfrecord"
write_tfrecord(in_path,out_path,args.data,"vocab_all",args.tokenizer)

print("Writing evaluation dataset...")
in_path = info["path"]+"/eval.csv"
out_path = info["path"]+"/eval_all.tfrecord"
write_tfrecord(in_path,out_path,args.data,"vocab_all",args.tokenizer)

print("Writing Test dataset...")
in_path = info["path"]+"/test.csv"
out_path = info["path"]+"/test_all.tfrecord"
write_tfrecord(in_path,out_path,args.data,"vocab_all",args.tokenizer)

