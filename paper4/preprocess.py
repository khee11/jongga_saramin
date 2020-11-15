#glove 및 vocab만들고 이를 저장 #, 그리고 tf.data.Dataset까지 만들고 return하는 모듈

from utils import *
import argparse
import os

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",type=str,choices=["dbpedia","agnews","yahoo","yelp"])
    parser.add_argument("--tokenizer",type=str,choices=["word","tweet"])
    parser.add_argument("--remove_first_row",action="store_true",help="Use if the file include column-header")
    parser.add_argument("--seed",type=int,default=1234)
    parser.add_argument("--num_label",type=int,help="per-class labeled examples")
    parser.add_argument("--num_eval",type=int,help="per-class evaluation examples")
    parser.add_argument("--num_unlab",type=int,help="per-class evaluation examples")
    parser.add_argument("--min_freq",type=int,help="Word whose frequency less or same with this are excluded")
    args = parser.parse_args()
    
    # Create some necessary folders
    os.makedirs(f"pretrained/{args.data}",exist_ok=True)

    if args.remove_first_row:
        remove_title(args.data)
        
    print("\nSplit the train data into the labeled, the unlabeled and evaluation ")
    split_whole_data(args.data,args.num_label,args.num_unlab,args.num_eval,args.seed)
    
    print("\nBuild the vocabulary from corpus")
    vocab_all_path = f"pretrained/{args.data}/vocab_all.txt"
    vectors_all_path = f"pretrained/{args.data}/vectors_all.npy"
    
    vocab_lab_path = f"pretrained/{args.data}/vocab_lab.txt"
    vectors_lab_path = f"pretrained/{args.data}/vectors_lab.npy"
    
    corpus_lab = build_corpus_from_csv(args.data,args.tokenizer,True,False)
    corpus_all = build_corpus_from_csv(args.data,args.tokenizer,False,False)
    
    save_vocab_from_corpus(corpus_lab,vocab_lab_path,0)
    save_vocab_from_corpus(corpus_all,vocab_all_path,args.min_freq)
    
    vocab_lab = load_vocab_from_txt(vocab_lab_path)
    vocab_all = load_vocab_from_txt(vocab_all_path)
    print("Match pretrained glove with our vocabulary...")
    match_vectors_with_vocab_from_txt(f"pretrained/glove/glove.840B.300d.txt",
        vocab_lab,300,vectors_lab_path)
    match_vectors_with_vocab_from_txt(f"pretrained/glove/glove.840B.300d.txt",
        vocab_all,300,vectors_all_path)
    
