import tensorflow as tf
import tensorflow_hub as hub
import os,sys,shutil
import numpy as np

import model_builder3 as builder
from data_loader import data_loader
"""
Assume data_loader.py has been run , so that data is ready to be used
"""
tf.get_logger().setLevel('INFO')
BATCH_SIZE = 64

dataset_train = data_loader("tfrecord_train")
dataset_train = dataset_train.repeat().shuffle(10000).batch(BATCH_SIZE)

dataset_eval = data_loader("tfrecord_eval")
dataset_eval = dataset_eval.repeat(1).batch(BATCH_SIZE)

iterator = tf.data.Iterator.from_structure(dataset_train.output_types,dataset_train.output_shapes)
train_init_op = iterator.make_initializer(dataset_train)
eval_init_op = iterator.make_initializer(dataset_eval)

image,text_ids,label,fname = iterator.get_next()

TEMP=0.5
m=builder.Bai_Model(image,text_ids,label,fname,TEMP)

# Set configuration for Session and training and saver and summary operation
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

# Some predefined value for training the model
MAX_EPOCHS = 40
TRAIN_SIZE= 5246
EVAL_SIZE=1310
TEST_SIZE = 1633


STEPS_PER_EPOCH = int(TRAIN_SIZE/BATCH_SIZE)+1
EVAL_STEPS_PER_EPOCH = int(EVAL_SIZE/BATCH_SIZE)+1
MAX_STEPS = MAX_EPOCHS*STEPS_PER_EPOCH
MODEL_PATH= os.path.join(os.getcwd(),"MODEL")
CKPT_PATH=os.path.join(os.getcwd(),"MODEL/model.ckpt")
# Checkpoint file, if "./abcd", then abcd file is created (with default suffix)
# On otherhand, latest_checkpoint(PATH) just tries to find the latest checkpoint files in the PATH directory

tf.summary.scalar("loss",m.loss)
tf.summary.scalar("learning_rate",m.lr)
summary_op=tf.summary.merge_all()
summary_writer=tf.summary.FileWriter(CKPT_PATH)
summary_writer_eval=tf.summary.FileWriter(CKPT_PATH)

# Variable except pretrained inception net to be saved
# This may store embedding,fusion part, and some fully connected layers
save_target = [var for var in tf.global_variables() if not var.name.startswith("module")]
saver = tf.train.Saver(save_target,max_to_keep=1)

earlyStop = np.zeros(shape=[MAX_EPOCHS,2]).astype(np.float32)
STOP_THRESHOLD = int(MAX_STEPS//2)
#########################
with tf.Session(config=config) as sess:
    with tf.variable_scope("integating", reuse=tf.AUTO_REUSE) as scope:
        sess.run(tf.global_variables_initializer())
        try:
            saver.restore(sess,tf.train.latest_checkpoint(MODEL_PATH))
        except:
            print("*****NO checkpoints to be restored*****")
        
        # train_handle = sess.run(dataset_train.string_handle())
        sess.run(train_init_op)
        epoch = int(1)
        for _ in range(MAX_STEPS):
            
            # Train through the whole data
            bs,_,loss_print,train_accuracy,current_lr,summaries,c_step =sess.run(
                            [m.bs,m.train_step,m.loss,m.accuracy,m.lr,summary_op,m.global_step],
                            feed_dict={m.init_lr:1e-2,m.rate:0.3})
            summary_writer.add_summary(summaries,c_step)
            if c_step%STEPS_PER_EPOCH==0:
                print("Epoch:{},size:{}, loss:{:.4f}, acc:{}, lr:{}".format(c_step//STEPS_PER_EPOCH,bs,loss_print,train_accuracy,current_lr))
            
            if c_step % STEPS_PER_EPOCH == 0:
                total_loss_eval=0.0
                total_loss_acc=0.0
                total_size=0
                saver.save(sess,CKPT_PATH,c_step)
                sess.run(eval_init_op)
                for _ in range(EVAL_STEPS_PER_EPOCH):
                    bs,loss_print,train_accuracy=sess.run(
                                [m.bs,m.loss,m.accuracy],
                                feed_dict={m.rate:0.0})
                    total_loss_eval+=loss_print*bs
                    total_loss_acc+=train_accuracy*bs
                    total_size += int(bs)
                total_loss_eval /= total_size
                total_loss_acc /= total_size
                print("    Evaluation({}) : {:.4f} {:6f}".format(total_size,total_loss_eval,total_loss_acc))
                if os.path.exists(m._FINE_DIR):
                    shutil.rmtree(m._FINE_DIR)
                    m.module.export(m._FINE_DIR,sess)
                else:
                    m.module.export(m._FINE_DIR,sess)
                
                earlyStop[(epoch-1),:]=[total_loss_eval,total_loss_acc]
                if c_step > STOP_THRESHOLD  :
                    is_stop = earlyStop[(epoch-4):(epoch-1),:]-earlyStop[(epoch-5):(epoch-2),:]
                    is_stop_loss = np.allclose(is_stop[:,0],np.zeros(shape=[3,],dtype=np.float32),atol=2e-2)
                    is_stop_acc = np.allclose(is_stop[:,1],np.zeros(shape=[3,],dtype=np.float32),atol=2e-3)
                    if is_stop_loss and is_stop_acc :
                        print("At {}, earlyStopping called".format(epoch))
                        break
                epoch +=int(1)
                sess.run(train_init_op)
                
                
                
