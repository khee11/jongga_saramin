# https://bskyvision.com/504
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import os,sys,shutil

class Bae_model:
    
    def __init__(self,img,txt,label):
        self.is_train=True
        EMBD_DIM=200
        LAMBDA=0.01
        EMBD_DIR="pretrained_skipgram.npy"
        self.img_input = img
        self.txt_input = txt
        self.label = label
        #self.img_input = tf.placeholder(tf.float32,[None,224,224,3])
        #self.txt_input = tf.placeholder(tf.int32,[None,MAX_LEN])
        #self.label = tf.placeholder(tf.float32,[None,NUM_CLASS])
        
        self.rate=tf.placeholder(dtype=tf.float32) 
        self.global_step = tf.Variable(0,trainable=False,name="global_step")
        with tf.control_dependencies([self.global_step]):
            gs_tensor=tf.cast(tf.identity(self.global_step),dtype=tf.float32)
        self.init_lr = tf.placeholder(tf.float32,[])
        
        def custom_lr(init_lr,global_step):
            if global_step<1000:
                return init_lr/20
            elif global_step < 2000:
                return init_lr/10
            elif global_step < 3000:
                return init_lr/5
            elif global_step < 7000:
                return init_lr/2
            else:
                return init_lr/(4)

        self.lr=tf.py_function(custom_lr,[self.init_lr,gs_tensor],tf.float32)
        
        final_img = tf.map_fn(tf.image.per_image_standardization,self.img_input)
        final_img = tf.cast(final_img,tf.float32)
        embedding_value = np.load(EMBD_DIR)
        embedding=tf.get_variable("embedding",shape=embedding_value.shape,dtype=tf.float32,
                       initializer=tf.constant_initializer(embedding_value))
        embedded_value=tf.nn.embedding_lookup(embedding,self.txt_input)
        
        module = tf.keras.applications.VGG16(include_top=False,pooling="avg")
        
        vgg_output=module(final_img)
        vgg_output = tf.layers.flatten(vgg_output)
        vgg_output = tf.layers.dense(vgg_output,4096)
        vgg_output=tf.nn.dropout(vgg_output,rate=self.rate)
        
        img_feature_reshape=tf.reshape(vgg_output,[-1,16,1,256])
        bn=tf.layers.BatchNormalization(trainable=is_train)
        img_feature_reshape=bn(img_feature_reshape)
        conv_x= tf.layers.conv2d(img_feature_reshape,256,[14,1],activation='relu')
        conv_x = tf.nn.dropout(conv_x,self.rate)
        max_x=tf.layers.max_pooling2d(conv_x,[2,1],[2,2])
        
        embedded_value_re = tf.reshape(embedded_value,[-1,MAX_LEN,EMBD_DIM,1])
        
        conv_0 = tf.layers.conv2d(embedded_value_re,NUM_FILTERS,[2,EMBD_DIM],activation='relu')
        bn0=tf.layers.BatchNormalization(trainable=is_train)
        conv_0 = bn0(conv_0)
        conv_0 = tf.nn.dropout(conv_0,self.rate)
        
        conv_1 = tf.layers.conv2d(embedded_value_re,NUM_FILTERS,[3,EMBD_DIM],activation='relu')
        bn1=tf.layers.BatchNormalization(trainable=is_train)
        conv_1 = bn1(conv_1)
        conv_1 = tf.nn.dropout(conv_1,self.rate)
        
        conv_2 = tf.layers.conv2d(embedded_value_re,NUM_FILTERS,[4,EMBD_DIM],activation='relu')
        bn2=tf.layers.BatchNormalization(trainable=is_train)
        conv_2 = bn2(conv_2)
        conv_2 = tf.nn.dropout(conv_2,self.rate)
        
        max_pool_0=tf.layers.max_pooling2d(conv_0,[MAX_LEN-2+1,1], [1,1])
        max_pool_1=tf.layers.max_pooling2d(conv_1,[MAX_LEN-3+1,1], (1,1))
        max_pool_2=tf.layers.max_pooling2d(conv_2,[MAX_LEN-4+1,1], (1,1))
        
        concat1 = tf.concat([max_pool_0,max_x],axis=1)
        concat2 = tf.concat([max_pool_1,max_x],axis=1)
        concat3 = tf.concat([max_pool_2,max_x],axis=1)
        concat4 = tf.concat([concat1,concat2,concat3],axis=1)
        
        conv_final = tf.layers.conv2d(concat4,512,[5,1],activation='relu')
        bn_final=tf.layers.BatchNormalization(trainable=is_train)
        conv_final = bn_final(conv_final)
        conv_final_maxpool=tf.layers.max_pooling2d(conv_final,[2,1],[2,2])
        conv_final_flat = tf.layers.flatten(conv_final_maxpool)
        conv_final_flat = tf.nn.dropout(conv_final_flat,self.rate)
        logits_final = tf.layers.dense(conv_final_flat,102,activation="softmax",
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(LAMBDA))
        
        prediction = tf.nn.softmax(logits_final)
        
        self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                        labels=self.label, logits=logits_final))
        
        self.train_step = tf.train.AdagradOptimizer(self.lr).minimize(
            self.loss,global_step=self.global_step)
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

if __name__=="__main__":
    DIR = "./BAE_MODEL/model.ckpt"
    model =Bae_model()    
    saver= tf.train.Saver()
    summary_writer = tf.summary.FileWriter(DIR)
    summary_op = tf.summary.merge_all()
    
    sess =tf.Session()
    sess.run(tf.global_variables_initializer())
    #summaries = sess.run([summary_op])
    saver.save(sess,DIR)
    #summary_writer.add_summary(summaries)
