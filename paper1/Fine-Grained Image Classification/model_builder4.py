
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os,shutil

"""
We use tensorflow for building Bai et al.,2018 model
In order to use Inception pre-trained model, we are using tensorflow_hub
And save it as SavedFormat, not session, since it looks like more lightweight
"""
class Bai_Model(object):
    
    def __init__(self,TEMP):
        self._URL= None
        self._LOCAL_DIR= None
        self._FINE_DIR= None
        self._EMBD_DIR= None
        self.TEMP = TEMP
        
        # Build input placeholder(Or can use tfrecord)
        self.img_input = None
        self.txt_input = None
        self.label_input = None
        self.fname= None
        self.bs = None
    
    def dir_setter(self,url,local,fine,embd):
        self._URL= url
        self._LOCAL_DIR= local
        self._FINE_DIR= fine
        self._EMBD_DIR= embd
    
    def build_input(self,image,text_ids,label,fname):
        self.img_input = image
        self.txt_input = text_ids
        self.label_input = label
        self.fname= fname
        self.bs = tf.shape(image)[0]
    
    def build_graph(self):
        # Droprate and global_step, initiali learning rate placeholder
        # gs_tensor is to be used in our custom learning rate function
        self.rate=tf.placeholder(dtype=tf.float32) 
        self.global_step = tf.Variable(0,trainable=False,name="global_step")
        with tf.control_dependencies([self.global_step]):
            gs_tensor=tf.cast(tf.identity(self.global_step),dtype=tf.float32)
        self.init_lr = tf.placeholder(tf.float32,[])    
        
        def custom_lr(init_lr,global_step):
            if global_step<400:
                return init_lr/20
            elif global_step<800:
                return init_lr/2
            elif global_step<2000:
                return init_lr
            else:
                return init_lr/4
                
        self.lr=tf.py_function(custom_lr,[self.init_lr,gs_tensor],tf.float32)
        
        # Input preprocessing: resize,normalized for images and word embedding for text
        # NOTE that embedding is the pretrained one
        img_resized_input = tf.image.resize(self.img_input, (299,299))
        final_img = tf.map_fn(tf.image.per_image_standardization,img_resized_input)
        embedding_value = np.load(self._EMBD_DIR)
        embedding=tf.get_variable("embedding",shape=embedding_value.shape,dtype=tf.float32,
                       initializer=tf.constant_initializer(embedding_value))
        embedded_value=tf.nn.embedding_lookup(embedding,self.txt_input) # [None,max_len,embed_dim]

    
        def inception_loader(url,local_dir,fine_dir,is_train=True):
            """
            Load Inception net from tensorflow hub URL, local file or fine-tuned file
            Each Argument should be a string representing the directory
            URL,Local directroy and fine-tuned model directory
            NOTE that fine-tuned model has its precedence over original model
            And is_train denotes whether the loaded model is meant to be trained or freezed
            """
            if os.path.exists(local_dir) and os.path.exists(fine_dir):
                module=hub.Module(fine_dir,trainable=is_train)
                print("\n\r*****Import from fine-tuned*****")
                self._from = "FINE"
            elif os.path.exists(local_dir):
                module=hub.Module("./Inception_v2",trainable=is_train)
                print("\n\r*****Import from original(local)*****")
                self._from = "LOCAL"
            else:
                module = hub.Module(url,trainable=is_train)
                print("\n\r*****Import from original(tf-hub)*****")
                self._from = "URL"
        
            return module
        
        # Load pre-trained Inception Net
        # By default, load the fined-tuned inception net from local file if exists
        self.module=inception_loader(self._URL,self._LOCAL_DIR,self._FINE_DIR)
        
        # Pass images to Inception net and get the result
        incep_output=self.module(final_img)
        incep_output=tf.nn.dropout(incep_output,rate=self.rate)
        self.incep_logits=tf.layers.dense(incep_output,1024)
        incep_logits_dropout = tf.nn.dropout(self.incep_logits,rate=self.rate)


        def Integrating(img_input,txt_input):
            """
            This is a fusion part(image features and text features)
            img_input = [bn, 1024]
            txt_input = [bn,Nmax,embedding_dim]
            """
            #val = np.ones((1024,300))
            #attn_matr = tf.keras.backend.variable(value=val, dtype='float64', name='attention_matrix', constraint=None)
            attn_matr= tf.get_variable("attn_matr",shape=[1024,200],dtype=tf.float32) 
            attn_weight = tf.matmul(img_input,attn_matr) # [bs,1024]*[1024,300]=[bs,300]
            txt_input=tf.transpose(txt_input,[0,2,1]) # [bs,embed_dim,Nmax]
            attn_weight = tf.einsum("ai,aij->aj",attn_weight,txt_input) # [bs,Nmax]
            attn_weight = tf.nn.softmax(tf.div(attn_weight,self.TEMP),axis=-1) # [bs,Nmax]
            augmented_text = tf.einsum("ai,aji->aj",attn_weight,txt_input)
            # final_feature = tf.concat([img_input,attn_weight],1)
            img_reduct = tf.layers.dense(img_input,512)
            fuse_feature = tf.concat([img_reduct,augmented_text],1)
            return attn_weight,fuse_feature
        
        # Fusion image and textual features
        self.attn_weight,self.fuse_feature =Integrating(incep_logits_dropout,embedded_value)

        
        # Logit and prediction layer
        classification= tf.get_variable("classification",shape=[712,102],dtype=tf.float32) 
        logits = tf.matmul(self.fuse_feature,classification) #batch
        logits= tf.nn.dropout(logits,rate=self.rate) # dropout 1
        self.pred= tf.nn.softmax(logits)
        self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                        labels=self.label_input, logits=logits)
               )
        # Train operation does not seem to have a significant effect, 
        # but stability during training is considered, so Adagrad is used
        self.train_step = tf.train.AdagradOptimizer(self.lr).minimize(
            self.loss,global_step=self.global_step)
        correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.label_input, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


if __name__=="__main__":

    URL = "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1"
    LOCAL_DIR=os.path.join(os.getcwd(),"Inception_v2")
    FINE_DIR=os.path.join(os.getcwd(),"./Inception_v2_fine")
    EMBD_DIR=os.path.join(os.getcwd(),"pretrained_skipgram.npy")
    
    from data_loader import data_loader
    print("Running model_builder with `main` will create a local directory of pretrained VGG16")
    dataset=data_loader("./tfrecord_data/tfrecord_eval")
    dataset = dataset.repeat(1).batch(1)
    iterator = tf.data.Iterator.from_structure(dataset.output_types,dataset.output_shapes)
    dataset_init_op = iterator.make_initializer(dataset)
    img,txt,label,fname = iterator.get_next()
    
    model=Bai_Model(URL,LOCAL_DIR,FINE_DIR,EMBD_DIR)
    model.dir_setter()
    model.build_input(img,txt,label,fname)
    model.build_graph()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("Save pretrained Inception net into a local directory...")
        model.module.export(model._LOCAL_DIR,sess)
        
