import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tqdm import tqdm
from utils import *
from tf_utils import *
import model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import losses,optimizers,metrics
import numpy as np

"""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data",type=str,default="yahoo") # "agnews","dbpedia","yahoo"
parser.add_argument("--model",type=str,default="cnn",choices=["cnn","lstm","swem_cat","swem_avg","swem_max","swem_hier"])
parser.add_argument("--is_mlp",type=bool,default=False)
parser.add_argument("--step",type=str,choices=["sup","at","vat","vadt","pi","both"],default="at")
parser.add_argument("--tokenizer",type=str,default="word")
parser.add_argument("--batch_size1",type=int,default=64) # for supervised in semi
parser.add_argument("--batch_size2",type=int,default=256) # for unsupervised in semi
parser.add_argument("--buffer_size1",type=int,default=1200) 
parser.add_argument("--buffer_size2",type=int,default=24000) 
parser.add_argument("--epochs",type=int,default=20)
parser.add_argument("--optimizer",type=str,default="Adam")
parser.add_argument("--lr",type=float,default=5e-4) # [1e-3, 3e-4, 2e-4, 1e-5] : for me, [3e-4,5e-4,1e-3]
parser.add_argument("--clip",type=float,default=-1.0)
parser.add_argument("--eta",type=float,default=2.0)
args = parser.parse_args()

if args.step =="sup" or args.step =="at":
    suffix = "lab"
    mode = "part"
else:
    suffix = "all"
    mode = "semi"
        
dataset_trn = build_dataset_from_tfrecord(mode,suffix,args.data,args.batch_size1,args.batch_size2\
        ,args.buffer_size1,args.buffer_size2,False)
dataset_eval = build_dataset_from_tfrecord("eval",suffix,args.data,128,None,None,None,False)
dataset_test = build_dataset_from_tfrecord("test",suffix,args.data,128,None,None,None,False)

global_max = 0
for batch in dataset_test:
    max_val = tf.reduce_max(batch["lab"]["x"])
    if global_max < max_val:
        global_max = max_val
global_max

"""

def setup_train(args):
    info = info_return(args.data)
    
    if args.step=="at" or args.step=="sup":
        suffix = "lab"
    else:
        suffix = "all"
    
    vocab = load_vocab_from_txt(f"pretrained/{args.data}/vocab_{suffix}.txt")
    vocab_size = len(vocab)
    embed_dim = 300
    vector_path =  f"pretrained/{args.data}/vectors_{suffix}.npy"
    vectors = load_embedding_matrix(vector_path ,True)
    
    # This model argument can be set by argparse, passing some 
    # unecessary ones (stride,context,.etc) default values
    model_kwargs = load_json("model_config.json",args.model)
    model_kwargs.update({
        "vocab_size" : vocab_size,
        "embed_dim" : embed_dim,
        "num_class" : info["num_class"],
        "embedding_path" : vectors,
        "is_mlp" : args.is_mlp
    })
    
    target_model = getattr(model,args.model)
    model_inst = target_model(**model_kwargs)
    
    if info["num_class"] == 2:
        loss_func = losses.BinaryCrossentropy(from_logits=True)
    elif info["num_class"] > 2:
        loss_func = losses.CategoricalCrossentropy(from_logits=True)

    if args.clip < 0.0:
        opt_kwargs = dict(learning_rate=args.lr) 
    else:
        opt_kwargs = dict(learning_rate=args.lr,clipnorm=args.clip)
    optimizer = getattr(optimizers,args.optimizer)
    opt_inst = optimizer(**opt_kwargs)
    
    if args.step == "sup":
        target_step = step
    elif args.step == "at":
        target_step = at_step
    elif args.step == "pi":
        target_step = pi_step
    elif args.step == "vat":
        target_step = vat_step
    elif args.step == "both":
        target_step = at_vat_step
    
    return vocab,model_inst,loss_func,opt_inst,target_step

def train(args):
    
    vocab,model_inst,loss_func,opt_inst,target_step = setup_train(args)
    
    if args.step =="sup" or args.step =="at":
        suffix = "lab"
        mode = "part"
    else:
        suffix = "all"
        mode = "semi"
        
    dataset_trn = build_dataset_from_tfrecord(mode,suffix,args.data,args.batch_size1,args.batch_size2\
        ,args.buffer_size1,args.buffer_size2,False)
    dataset_eval = build_dataset_from_tfrecord("eval",suffix,args.data,128,None,None,None,False)
    dataset_test = build_dataset_from_tfrecord("test",suffix,args.data,128,None,None,None,False)
    
    ### custom train step 부분
    trn_loss_metric = metrics.Mean() # Train loss
    trn_acc_metric = metrics.Accuracy() # Train Accuracy
    
    ckpt_dir = f"model_log/{args.data}/{args.step}/{args.model}"
    os.makedirs(ckpt_dir,exist_ok=True)
    log_dir = ckpt_dir + "/train_log.csv"
    logger_pipe = open(log_dir,"w")
    logger_pipe.write("epoch,train_loss,train_acc,eval_loss,eval_acc\n")
    ckpt_manager =  tf.train.CheckpointManager(tf.train.Checkpoint(model=model_inst)\
        ,directory=ckpt_dir,max_to_keep=1)

    eval_acc_best = -1e-1
    eval_loss_best = 1e10
    
    for epoch in range(args.epochs):
        pbar = tqdm(dataset_trn)
        for batch in pbar:
            
            if args.step == "sup":
                func_args = [model_inst,opt_inst,loss_func,batch]
            elif args.step == "pi":
                weight = lin_weight(args.start_weight,args.start_weight,0.05,0,epoch)
                func_args = [model_inst,opt_inst,loss_func,batch,weight]            
            elif args.step == "at" or args.step == "vat" or args.step == "both" :
                weight = lin_weight(args.start_weight,args.start_weight,0.05,0,epoch)
                func_args = [model_inst,opt_inst,loss_func,batch,args.eta,weight]            

            loss_val,logit,gs = target_step(*func_args)
            
            trn_loss_metric.update_state(loss_val) # metrics for training
            trn_acc_metric.update_state(tf.argmax(batch["lab"]["y"],1),tf.argmax(logit,1))
            current_loss = trn_loss_metric.result().numpy()
            current_acc = trn_acc_metric.result().numpy()
            current_step = gs.numpy()
            pbar.set_description(f"Epoch {epoch}, Step {current_step:d}, Loss {current_loss:.5f}, Acc {current_acc:.5f}")

        trn_loss_metric.reset_states()
        trn_acc_metric.reset_states()
          
        eval_loss,eval_acc= evaluate(model_inst,dataset_eval,loss_func)
        print("------[Eval] Loss {:.5f}, Acc {:.5f}".format(eval_loss,eval_acc))
        
        if eval_acc_best < eval_acc:
            ckpt_manager.save(epoch)
            eval_acc_best = eval_acc
            eval_loss_best = eval_loss
        elif eval_acc_best == eval_acc:
            if eval_loss_best < eval_loss:
                pass
            else :
                ckpt_manager.save(epoch)
                eval_acc_best = eval_acc
                eval_loss_best = eval_loss
        # https://docs.python.org/3/tutorial/inputoutput.html#formatted-string-literals        
        logger_pipe.write(f"{epoch!s},{current_loss!s},{current_acc!s},{eval_loss!s},{eval_acc!s}\n") 

    ckpt_manager.restore_or_initialize()
    test_loss,test_acc = evaluate(model_inst,dataset_test,loss_func)
    print("On test set, Loss {:.5f}, Acc {:.5f}".format(test_loss,test_acc))
    logger_pipe.write(f"OnTest,-,-,{test_loss!s},{test_acc!s}\n")
    logger_pipe.close()
    return 0

if __name__=="__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",type=str,default="yahoo") # "agnews","dbpedia","yahoo"
    parser.add_argument("--model",type=str,default="cnn",choices=["lstm","lstm_base","cnn","swem_cat","swem_hier","swem_max","swem_avg"])
    parser.add_argument("--is_mlp",type=bool,default=False)
    parser.add_argument("--step",type=str,choices
    =["sup","at","vat","pi","both"],default="pi")
    parser.add_argument("--tokenizer",type=str,default="word")
    parser.add_argument("--batch_size1",type=int,default=64) # for supervised in semi
    parser.add_argument("--batch_size2",type=int,default=256) # for unsupervised in semi
    parser.add_argument("--buffer_size1",type=int,default=1400)
    parser.add_argument("--buffer_size2",type=int,default=28000)
    parser.add_argument("--epochs",type=int,default=30) # 100 for swem_avg, 75 for swem_cat,swem_avg,swem_max , 50 for swems' semisupervised, 30 for non-swem semi-supervised learning 
    parser.add_argument("--optimizer",type=str,default="Adam")
    parser.add_argument("--lr",type=float,default=1e-3) # for me, [3e-4,5e-4,1e-3,3e-3]
    parser.add_argument("--clip",type=float,default=-1.0) # minus for not use clipping gradients
    parser.add_argument("--eta",type=float,default=2.0) # size of norm for embeddign noise
    parser.add_argument("--start_weight",type=float,default=0.35) 
    parser.add_argument("--end_weight",type=float,default=0.50)
    args = parser.parse_args()
    
    print("\n\nTrain {} ({}), on {}\n\n".format(args.model,args.step,args.data))
    train(args)
