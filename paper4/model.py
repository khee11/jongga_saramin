import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,losses,optimizers,activations

class cnn(keras.Model):

    def __init__(self,**kwargs):
        super(cnn,self).__init__()
        
        if kwargs["embedding_path"] is None:
            embedding_initializer = "uniform"
        elif type(kwargs["embedding_path"]) == str:
            embedding_initializer = np.load(embedding_path).astype(np.float32)
            embedding_initializer = tf.keras.initializers.Constant(embedding_initializer)
            print("Initialize from",embedding_path)
        elif type(kwargs["embedding_path"]) == np.ndarray :
            embedding_initializer = tf.keras.initializers.Constant(kwargs["embedding_path"])
            print("Initialize from a numpy array")
        
        if kwargs["is_mlp"]:
            self.embedding = keras.Sequential([
                layers.Embedding(kwargs["vocab_size"],kwargs["embed_dim"],embedding_initializer, mask_zero=True),
                layers.Dense(kwargs["embed_dim"],"relu")
            ])
        else:
            self.embedding = layers.Embedding(kwargs["vocab_size"],kwargs["embed_dim"],embedding_initializer, mask_zero=True)
        
        self.drop = layers.Dropout(kwargs["drop_rate"])
        self.conv = layers.Conv1D(kwargs["num_kernel"],kwargs["context_size"],kwargs["stride"])
        self.drop2 = layers.Dropout(kwargs["drop_rate"])
        self.fc = layers.Dense(kwargs["fc_dim"])
        self.classifier = layers.Dense(kwargs["num_class"])

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None,None],dtype=tf.int32)
    ])
    def embedder(self,inp):
        return self.embedding(inp)
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None,None,300],dtype=tf.float32),
        tf.TensorSpec(shape=[],dtype=tf.bool)])     
    def call(self,x,training):
        out = self.conv(x)
        out = tf.reduce_max(out,1) # if return_sequence = True
        out = self.drop(out,training)
        out = self.fc(out)
        out = self.drop2(out,training)
        logit = self.classifier(out)
        return logit

class lstm_base(keras.Model):
    
    def __init__(self,**kwargs):
        super(lstm_base,self).__init__()
        
        if kwargs["embedding_path"] is None:
            embedding_initializer = "uniform"
        elif type(kwargs["embedding_path"]) == str:
            embedding_initializer = np.load(embedding_path).astype(np.float32)
            embedding_initializer = tf.keras.initializers.Constant(embedding_initializer)
            print("Initialize from",embedding_path)
        elif type(kwargs["embedding_path"]) == np.ndarray :
            embedding_initializer = tf.keras.initializers.Constant(kwargs["embedding_path"])
            print("Initialize from a numpy array")
        
        if kwargs["is_mlp"]:
            self.embedding = keras.Sequential([
                layers.Embedding(kwargs["vocab_size"],kwargs["embed_dim"],embedding_initializer, mask_zero=True),
                layers.Dense(kwargs["embed_dim"],"relu")
            ])
        else:
            self.embedding = layers.Embedding(kwargs["vocab_size"],kwargs["embed_dim"],embedding_initializer, mask_zero=True)
        
        self.lstm = layers.Bidirectional(layers.LSTM(kwargs["hidden_dim"],return_sequences=False))
        self.drop = layers.Dropout(kwargs["drop_rate"])
        self.drop2 = layers.Dropout(kwargs["drop_rate"])
        self.fc = layers.Dense(kwargs["fc_dim"])
        self.classifier = layers.Dense(kwargs["num_class"])

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None,None],dtype=tf.int32)
    ])
    def embedder(self,inp):
        return self.embedding(inp)
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None,None,300],dtype=tf.float32),
        tf.TensorSpec(shape=[],dtype=tf.bool)])      
    def call(self,x,training):
        out = self.lstm(x)
        out = self.drop(out,training)
        out = self.fc(out)
        out = self.drop2(out,training)
        logit = self.classifier(out)
        return logit

class lstm(keras.Model):

    def __init__(self,**kwargs):
        super(lstm,self).__init__()
        
        if kwargs["embedding_path"] is None:
            embedding_initializer = "uniform"
        elif type(kwargs["embedding_path"]) == str:
            embedding_initializer = np.load(embedding_path).astype(np.float32)
            embedding_initializer = tf.keras.initializers.Constant(embedding_initializer)
            print("Initialize from",embedding_path)
        elif type(kwargs["embedding_path"]) == np.ndarray :
            embedding_initializer = tf.keras.initializers.Constant(kwargs["embedding_path"])
            print("Initialize from a numpy array")
        
        if kwargs["is_mlp"]:
            self.embedding = keras.Sequential([
                layers.Embedding(kwargs["vocab_size"],kwargs["embed_dim"],embedding_initializer, mask_zero=True),
                layers.Dense(kwargs["embed_dim"],"relu")
            ])
        else:
            self.embedding = layers.Embedding(kwargs["vocab_size"],kwargs["embed_dim"],embedding_initializer, mask_zero=True)
        
        self.lstm = layers.Bidirectional(layers.LSTM(kwargs["hidden_dim"],return_sequences=True))
        self.drop = layers.Dropout(kwargs["drop_rate"])
        self.drop2 = layers.Dropout(kwargs["drop_rate"])
        self.fc = layers.Dense(kwargs["fc_dim"])
        self.classifier = layers.Dense(kwargs["num_class"])
        
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None,None],dtype=tf.int32)
    ])
    def embedder(self,inp):
        return self.embedding(inp)
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None,None,300],dtype=tf.float32),
        tf.TensorSpec(shape=[],dtype=tf.bool)])      
    def call(self,x,training):
        out = self.lstm(x)
        out = tf.reduce_max(out,1) # if return_sequence = True
        out = self.drop(out,training)
        out = self.fc(out)
        out = self.drop2(out,training)
        logit = self.classifier(out)
        return logit

class swem_max(keras.Model):
    
    def __init__(self,**kwargs): 
        super(swem_max,self).__init__()
        
        if kwargs["embedding_path"] is None:
            embedding_initializer = "uniform"
        elif type(kwargs["embedding_path"]) == str:
            embedding_initializer = np.load(embedding_path).astype(np.float32)
            embedding_initializer = tf.keras.initializers.Constant(embedding_initializer)
            print("Initialize from",embedding_path)
        elif type(kwargs["embedding_path"]) == np.ndarray :
            embedding_initializer = tf.keras.initializers.Constant(kwargs["embedding_path"])
            print("Initialize from a numpy array")
        
        if kwargs["is_mlp"]:
            self.embedding = keras.Sequential([
                layers.Embedding(kwargs["vocab_size"],kwargs["embed_dim"],embedding_initializer, mask_zero=True),
                layers.Dense(kwargs["embed_dim"],"relu")
            ])
        else:
            self.embedding = layers.Embedding(kwargs["vocab_size"],kwargs["embed_dim"],embedding_initializer, mask_zero=True)    
        self.drop = layers.Dropout(kwargs["drop_rate"])
        self.fc = layers.Dense(kwargs["fc_dim"],"relu")
        self.drop2 = layers.Dropout(kwargs["drop_rate"])
        self.classifier = layers.Dense(kwargs["num_class"])

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None,None],dtype=tf.int32)
    ])
    def embedder(self,inp):
        return self.embedding(inp)
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None,None,300],dtype=tf.float32),
        tf.TensorSpec(shape=[],dtype=tf.bool)])         
    def call(self,x,training):
        _max = tf.reduce_max(x,1)
        out = self.drop(_max,training)
        out = self.fc(out)
        out = self.drop2(out,training)
        logit = self.classifier(out)
        return logit

class swem_avg(keras.Model):
    
    def __init__(self,**kwargs):
        super(swem_avg,self).__init__()
        if kwargs["embedding_path"] is None:
            embedding_initializer = "uniform"
        elif type(kwargs["embedding_path"]) == str:
            embedding_initializer = np.load(embedding_path).astype(np.float32)
            embedding_initializer = tf.keras.initializers.Constant(embedding_initializer)
            print("Initialize from",embedding_path)
        elif type(kwargs["embedding_path"]) == np.ndarray :
            embedding_initializer = tf.keras.initializers.Constant(kwargs["embedding_path"])
            print("Initialize from a numpy array")
        
        if kwargs["is_mlp"]:
            self.embedding = keras.Sequential([
                layers.Embedding(kwargs["vocab_size"],kwargs["embed_dim"],embedding_initializer, mask_zero=True),
                layers.Dense(kwargs["embed_dim"],"relu")
            ])
        else:
            self.embedding = layers.Embedding(kwargs["vocab_size"],kwargs["embed_dim"],embedding_initializer, mask_zero=True)
        self.drop = layers.Dropout(kwargs["drop_rate"])
        self.fc = layers.Dense(kwargs["fc_dim"],"relu")
        self.drop2 = layers.Dropout(kwargs["drop_rate"])
        self.classifier = layers.Dense(kwargs["num_class"])

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None,None],dtype=tf.int32)
    ])
    def embedder(self,inp):
        return self.embedding(inp)
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None,None,300],dtype=tf.float32),
        tf.TensorSpec(shape=[],dtype=tf.bool)])      
    def call(self,x,training):
        _avg = tf.reduce_mean(x,1)
        out = self.drop(_avg,training)
        out = self.fc(out)
        out = self.drop2(out,training)
        logit = self.classifier(out)
        return logit

class swem_hier(keras.Model):
    
    def __init__(self,**kwargs): #vocab_size,embed_dim,context_size,stride,drop_rate,fc_dim,num_class,embedding_path=None):
        super(swem_hier,self).__init__()
        if kwargs["embedding_path"] is None:
            embedding_initializer = "uniform"
        elif type(kwargs["embedding_path"]) == str:
            embedding_initializer = np.load(embedding_path).astype(np.float32)
            embedding_initializer = tf.keras.initializers.Constant(embedding_initializer)
            print("Initialize from",embedding_path)
        elif type(kwargs["embedding_path"]) == np.ndarray :
            embedding_initializer = tf.keras.initializers.Constant(kwargs["embedding_path"])
            print("Initialize from a numpy array")
        
        if kwargs["is_mlp"]:
            self.embedding = keras.Sequential([
                layers.Embedding(kwargs["vocab_size"],kwargs["embed_dim"],embedding_initializer, mask_zero=True),
                layers.Dense(kwargs["embed_dim"],"relu")
            ])
        else:
            self.embedding = layers.Embedding(kwargs["vocab_size"],kwargs["embed_dim"],embedding_initializer, mask_zero=True)    
        self.hier = layers.AveragePooling1D(pool_size=kwargs["context_size"],strides=kwargs["stride"])
        self.drop = layers.Dropout(kwargs["drop_rate"])
        self.fc = layers.Dense(kwargs["fc_dim"],"relu")
        self.drop2 = layers.Dropout(kwargs["drop_rate"])
        self.classifier = layers.Dense(kwargs["num_class"])

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None,None],dtype=tf.int32)
    ])
    def embedder(self,inp):
        return self.embedding(inp)
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None,None,300],dtype=tf.float32),
        tf.TensorSpec(shape=[],dtype=tf.bool)])     
    def call(self,x,training):
        out = self.hier(x)
        out = self.drop(out,training)
        out = tf.reduce_max(out,1)
        out = self.fc(out)
        out = self.drop2(out,training)
        logit = self.classifier(out)
        return logit
    
class swem_cat(keras.Model):
    
    def __init__(self,**kwargs):
        super(swem_cat,self).__init__()
        if kwargs["embedding_path"] is None:
            embedding_initializer = "uniform"
        elif type(kwargs["embedding_path"]) == str:
            embedding_initializer = np.load(embedding_path).astype(np.float32)
            embedding_initializer = tf.keras.initializers.Constant(embedding_initializer)
            print("Initialize from",embedding_path)
        elif type(kwargs["embedding_path"]) == np.ndarray :
            embedding_initializer = tf.keras.initializers.Constant(kwargs["embedding_path"])
            print("Initialize from a numpy array")
        
        if kwargs["is_mlp"]:
            self.embedding = keras.Sequential([
                layers.Embedding(kwargs["vocab_size"],kwargs["embed_dim"],embedding_initializer, mask_zero=True),
                layers.Dense(kwargs["embed_dim"],"relu")
            ])
        else:
            self.embedding = layers.Embedding(kwargs["vocab_size"],kwargs["embed_dim"],embedding_initializer, mask_zero=True)            
        
        self.drop = layers.Dropout(kwargs["drop_rate"])
        self.fc = layers.Dense(kwargs["fc_dim"],"relu")
        self.drop2 = layers.Dropout(kwargs["drop_rate"])
        self.classifier = layers.Dense(kwargs["num_class"])

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None,None,300],dtype=tf.float32),
        tf.TensorSpec(shape=[],dtype=tf.bool)])     
    def call(self,x,training):
        _avg = tf.reduce_mean(x,1)
        _max = tf.reduce_max(x,1)
        out = tf.concat([_avg,_max],axis=1)
        out = self.drop(out,training)
        out = self.fc(out)
        out = self.drop2(out,training)
        logit = self.classifier(out)
        return logit
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None,None],dtype=tf.int32)
    ])
    def embedder(self,inp):
        return self.embedding(inp)
