import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os,sys
import shutil
from PIL import Image

import model_builder3 as builder
from data_loader import data_loader,build_embedding_mat

BATCH_SIZE = 64
dataset_test = data_loader("tfrecord_test")
dataset_test = dataset_test.repeat(1).batch(BATCH_SIZE)

dataset_train = data_loader("tfrecord_train")
dataset_train = dataset_train.repeat(1).batch(BATCH_SIZE)

dataset_eval = data_loader("tfrecord_eval")
dataset_eval = dataset_eval.repeat(1).batch(BATCH_SIZE)

iterator = tf.data.Iterator.from_structure(dataset_train.output_types,dataset_train.output_shapes)
train_init_op = iterator.make_initializer(dataset_train)
eval_init_op = iterator.make_initializer(dataset_eval)
test_init_op = iterator.make_initializer(dataset_test)

image,text_ids,label,fname = iterator.get_next()

m=builder.Bai_Model(image,text_ids,label,fname)

# Set configuration for Session and training and saver and summary operation
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

save_target = [var for var in tf.global_variables() if not var.name.startswith("module")]
saver = tf.train.Saver(save_target,max_to_keep=1)

BATCH_SIZE = 64
TRAIN_SIZE= 5246
EVAL_SIZE=1310
TEST_SIZE = 1633
MODEL_PATH= os.path.join(os.getcwd(),"MODEL")
CKPT_PATH=os.path.join(os.getcwd(),"MODEL/model.ckpt")

STEPS_PER_EPOCH = int(TRAIN_SIZE/BATCH_SIZE)+1
EVAL_STEPS_PER_EPOCH = int(EVAL_SIZE/BATCH_SIZE)+1
TEST_STEPS_PER_EPOCH = int(TEST_SIZE/BATCH_SIZE)+1

#########################
with tf.Session(config=config) as sess:
    with tf.variable_scope("integating", reuse=tf.AUTO_REUSE) as scope:
        sess.run(tf.global_variables_initializer())
        try:
            saver.restore(sess,tf.train.latest_checkpoint(MODEL_PATH))
            print("*****Saved parameters are loaded*****")
        except:
            print("*****NO checkpoints to be restored*****")
        
        # For train set
        sess.run(train_init_op)
        train_loss =0.0
        train_acc =0.0
        train_size =0
        for _ in range(STEPS_PER_EPOCH):
            
            # Train through the whole data
            bs,loss_print,train_accuracy=sess.run(
                            [m.bs,m.loss,m.accuracy],feed_dict={m.rate:0.0})
            train_loss+=loss_print*bs
            train_acc+=train_accuracy*bs
            train_size+=bs
            
        print("Train : size:{}, loss:{:.4f}, acc:{}".format(train_size,train_loss/train_size,train_acc/train_size))
        
        # For eval
        sess.run(eval_init_op)
        eval_loss =0.0
        eval_acc =0.0
        eval_size =0
        for _ in range(EVAL_STEPS_PER_EPOCH):
            
            # Train through the whole data
            bs,loss_print,eval_accuracy=sess.run(
                            [m.bs,m.loss,m.accuracy],feed_dict={m.rate:0.0})
            eval_loss+=loss_print*bs
            eval_acc+=eval_accuracy*bs
            eval_size+=bs
            
        print("Eval : size:{}, loss:{:.4f}, acc:{}".format(eval_size,eval_loss/eval_size,eval_acc/eval_size))
        
        # For Test
        sess.run(test_init_op)
        test_loss =0.0
        test_acc =0.0
        test_size =0
        for _ in range(TEST_STEPS_PER_EPOCH):
            
            # Train through the whole data
            bs,loss_print,test_accuracy=sess.run(
                            [m.bs,m.loss,m.accuracy],feed_dict={m.rate:0.0})
            test_loss+=loss_print*bs
            test_acc+=test_accuracy*bs
            test_size+=bs
            
        print("Test : size:{}, loss:{:.4f}, acc:{}".format(test_size,test_loss/test_size,test_acc/test_size))
        
        
        # Some demonstration
        sess.run(test_init_op)
        img,txt,lbl,file_,attn_val = sess.run([image,text_ids,label,fname,m.attn_weight]
                                              ,feed_dict={m.rate:0.0})
        
        vocab,vocab_inv=build_embedding_mat(False)
        
        Image.fromarray(img[0]).show()
        print_text = [vocab_inv.get(id_,"개") for id_ in txt[0]]
        print("The corresponding text : \n\r {}".format(" ".join(print_text)))
        #print("Class : {}".format(lbl[0]))
        #print("File name : {}".format(file_[0].decode("utf-8")))
        print("Attention value(softmax) visualization:\n{}".format(attn_val[0]))
        idx=np.argsort(-attn_val[0])
        print(np.array(print_text)[idx])

"""
with tf.Session(config=config) as sess:
    with tf.variable_scope("integating", reuse=tf.AUTO_REUSE) as scope:
        sess.run(tf.global_variables_initializer())
        try:
            saver.restore(sess,tf.train.latest_checkpoint(MODEL_PATH))
            print("*****Saved parameters are loaded*****")
        except:
            print("*****NO checkpoints to be restored*****")
        
        sess.run(test_init_op)
        img,txt,lbl,file_,attn_val = sess.run([image,text_ids,label,fname,m.attn_weight]
                                              ,feed_dict={m.rate:0.0})
        Image.fromarray(img[0]).show()
        vocab,vocab_inv=build_embedding_mat(False)

        print_text = [vocab_inv.get(id_,"개") for id_ in txt[0]]
        print("The corresponding text : \n\r {}".format(" ".join(print_text)))
        print("Class : {}".format(lbl[0]))
        print("File name : {}".format(file_[0].decode("utf-8")))
        print("Attention value(softmax) visualization:\n{}".format(attn_val[0]))

Image.fromarray(img[1]).show()
idx=np.argsort(-attn_val[1])
print(print_text)
print(np.array(print_text)[idx])
"""
