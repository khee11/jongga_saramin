import tensorflow as tf
import numpy as np
from utils import *

mse_loss_func = tf.keras.losses.MeanSquaredError()
eps = 1e-6 # small value for differentiation

def build_dataset_from_tfrecord(mode,suffix,data_type,batch_size,batch_size2,buffer_size,buffer_size2,is_rpl):
        
    info = info_return(data_type)
    vocab = load_vocab_from_txt(f"pretrained/{data_type}/vocab_{suffix}.txt")
    vocab_size = len(vocab)
    
    description = {
        "tokens":tf.io.VarLenFeature(tf.int64),
        "label":tf.io.FixedLenFeature([],tf.int64)}
        
    def replace_tokens(tokens):
        new_tokens = np.copy(tokens)
        length = len(tokens)-1 # does not change <eos>
        change_num = int(length*0.1)
        new_idx = np.random.choice(np.arange(length),change_num,replace=False)
        #new_words = np.random.randint(3,length,change_num) # excep <unk>,<pad>,<eos>
        new_tokens[new_idx] = 1 # for unk
        return new_tokens
    
    def tf_replace_tokens(tokens):
        new_tok = tf.py_function(replace_tokens,[tokens],[tf.int32])
        return new_tok
    
    def parse_func(example):
        parsed = tf.io.parse_single_example(example,description)
        tokens = parsed["tokens"]
        tokens= tf.sparse.to_dense(tokens)
        label = tf.one_hot(parsed["label"],depth=info["num_class"])
        return {"lab":{"x":tf.cast(tokens,dtype=tf.int32),"y":label}}
    
    def parse_func_for_semi(example):
        parsed = tf.io.parse_single_example(example,description)
        tokens = parsed["tokens"]
        tokens= tf.sparse.to_dense(tokens)
        label = tf.one_hot(parsed["label"],depth=info["num_class"])
        return {"x":tf.cast(tokens,dtype=tf.int32),"y":label}
    
    def perturb_parse_func(example):
        parsed = tf.io.parse_single_example(example,description)
        tokens = parsed["tokens"]
        tokens= tf.sparse.to_dense(tokens)
        if is_rpl:
            new_tokens = tf.squeeze(tf_replace_tokens(tokens)) # NOTE that tokens are tensor
        else:
            new_tokens = tf.identity(tokens)
        return {"x1":tf.cast(tokens,dtype=tf.int32),"x2":tf.cast(new_tokens,dtype=tf.int32)}
        
    if mode=="test" or mode=="eval":
        path = info["path"]+f"/{mode}_{suffix}.tfrecord"
        ds = tf.data.TFRecordDataset(path)\
            .map(parse_func,num_parallel_calls=tf.data.experimental.AUTOTUNE)\
            .prefetch(tf.data.experimental.AUTOTUNE)\
            .padded_batch(batch_size,{"lab":{"x":[None],"y":[info["num_class"]]}})
    elif mode=="part":
        path = info["path"]+f"/train_{suffix}_lab.tfrecord"
        ds = tf.data.TFRecordDataset(path)\
            .shuffle(buffer_size)\
            .map(parse_func,num_parallel_calls=tf.data.experimental.AUTOTUNE)\
            .prefetch(tf.data.experimental.AUTOTUNE)\
            .padded_batch(batch_size,{"lab":{"x":[None],"y":[info["num_class"]]}})
    elif mode=="semi":
        path_lab = info["path"]+f"/train_{suffix}_lab.tfrecord"
        path_unlab = info["path"]+f"/train_{suffix}_unlab.tfrecord"
        ds_lab = tf.data.TFRecordDataset(path_lab)\
            .shuffle(buffer_size)\
            .map(parse_func_for_semi,num_parallel_calls=tf.data.experimental.AUTOTUNE)\
            .prefetch(tf.data.experimental.AUTOTUNE)\
            .padded_batch(batch_size,{"x":[None],"y":[info["num_class"]]})
        ds_unlab = tf.data.TFRecordDataset(path_unlab)\
            .shuffle(buffer_size2)\
            .map(perturb_parse_func,num_parallel_calls=tf.data.experimental.AUTOTUNE)\
            .prefetch(tf.data.experimental.AUTOTUNE)\
            .padded_batch(batch_size2,{"x1":[None],"x2":[None]})
        ds = tf.data.Dataset.zip({"lab":ds_lab,"unlab":ds_unlab})    
    return ds

def entropy_loss_func(logit):
    p = tf.nn.softmax(logit)
    return -tf.reduce_mean(tf.reduce_sum(p * tf.nn.log_softmax(logit), 1))

def kld_loss_func(logit1,logit2):
    p = tf.nn.softmax(logit1, -1)
    kld_loss = tf.reduce_mean(tf.reduce_sum(
        p * (tf.nn.log_softmax(logit1,-1)-tf.nn.log_softmax(logit2,-1)), 1)
    )
    return kld_loss

def get_normalized_vector(d):
    batch_size,length,dim = tf.shape(d)
    d = tf.reshape(d,[batch_size,-1])
    d /= (tf.reduce_max(d,1,keepdims=True)+1e-12)
    d /= (tf.math.sqrt(tf.reduce_sum(tf.pow(d,2.0),1,keepdims=True)+1e-6))
    d = tf.reshape(d,[batch_size,length,dim])
    return d

def exp_weight(init,end,factor,warmup,epoch):
    actual_pow = tf.cond(tf.less(epoch,warmup),lambda : 1.0, lambda :epoch-warmup)
    current_factor = tf.pow(factor,actual_pow)
    current_val = init*current_factor
    return_val = tf.cond(tf.less(current_val,end),lambda : current_val, lambda : end)
    return return_val

def lin_weight(init,end,factor,warmup,epoch):
    actual_pow = tf.cond(tf.less(epoch,warmup),lambda : 0.0, lambda :epoch-warmup)
    current_factor = factor*actual_pow
    current_val = init +current_factor
    return_val = tf.cond(tf.less(current_val,end),lambda : current_val, lambda : end)
    return return_val

def evaluate(model,dataset,loss_func):
    loss_metric = tf.keras.metrics.Mean()
    acc_metric = tf.keras.metrics.Accuracy()   
    for batch in dataset:
        embedded = model.embedder(batch["lab"]["x"])
        logit = model(embedded,False)
        pred = tf.argmax(logit,1)
        true_label =  tf.argmax(batch["lab"]["y"],1)
        loss_val = loss_func(batch["lab"]["y"],logit)
        loss_metric.update_state(loss_val)
        acc_metric.update_state(true_label,pred)
    return loss_metric.result().numpy(), acc_metric.result().numpy()

def step(model,optimizer,loss_func,batch):
    with tf.GradientTape() as tape:
        embedded = model.embedder(batch["lab"]["x"])
        logit = model(embedded,True)
        total_loss = loss_func(batch["lab"]["y"],logit)
    grad = tape.gradient(total_loss,model.trainable_variables)
    gs = optimizer.apply_gradients(zip(grad,model.trainable_variables))
    return total_loss,logit,gs

def pi_step(model,optimizer,loss_func,batch,weight=0.5):
    sup_w = 1.0 - weight
    with tf.GradientTape() as tape:
        embedded = model.embedder(batch["lab"]["x"])
        logit = model(embedded,True)
        sup_loss = loss_func(batch["lab"]["y"],logit)
        
        embedded1 = model.embedder(batch["unlab"]["x1"])
        logit1 = model(embedded1,True)
        embedded2 = model.embedder(batch["unlab"]["x2"])
        logit2 = model(embedded2,True)
        ent_loss = entropy_loss_func(logit1)
        mse_loss = mse_loss_func(tf.nn.softmax(logit1,-1),tf.nn.softmax(logit2,-1))
        total_loss = sup_w*sup_loss + weight*(ent_loss+mse_loss)
    grad = tape.gradient(total_loss,model.trainable_variables)
    gs = optimizer.apply_gradients(zip(grad,model.trainable_variables))
    return total_loss,logit,gs

def generate_at_perturbation(model,loss_func,x,y,eta):
    with tf.GradientTape() as tape:
        tape.watch(x)
        logit = model(x,True)
        ce_loss = loss_func(y,logit)
    grad= tape.gradient(ce_loss,x)
    return get_normalized_vector(grad)*eta

def at_step(model,optimizer,loss_func,batch,eta,weight=0.5):
    sup_w = 1.0 - weight
    with tf.GradientTape() as tape:
        embedded = model.embedder(batch["lab"]["x"])
        logit = model(embedded,True)
        with tape.stop_recording():
            r_at = generate_at_perturbation(model,loss_func,embedded,batch["lab"]["y"],eta)
        logit_at = model(embedded+r_at,True)
        total_loss = sup_w*loss_func(batch["lab"]["y"],logit) + weight*loss_func(batch["lab"]["y"],logit_at)
    grad = tape.gradient(total_loss,model.trainable_variables)
    gs = optimizer.apply_gradients(zip(grad,model.trainable_variables))
    return total_loss,logit,gs

def generate_vat_perturbation(model,x,logit,eta,num_iter=2):
    
    logit = tf.nn.softmax(logit)
    d = get_normalized_vector(tf.random.normal(shape=x.shape))
    
    for _ in range(num_iter):
        with tf.GradientTape() as tape:
            tape.watch([x,d])
            x_d = x + (eps*d)
            logit_vat = model(x_d,True)
            kld_loss_val = kld_loss_func(logit,logit_vat)
        d = tape.gradient(kld_loss_val,x_d)
        d = get_normalized_vector(d)
    r_vat = eta * d
    return r_vat

def vat_step(model,optimizer,loss_func,batch,eta,weight=0.5,num_iter=2):
    sup_w = 1.0 -weight
    
    with tf.GradientTape() as tape:
        embedded = model.embedder(batch["lab"]["x"])
        logit = model(embedded,True)
        sup_loss = loss_func(batch["lab"]["y"],logit)
        
        embedded_ul = model.embedder(batch["unlab"]["x1"])
        logit_ul = model(embedded_ul,True)
        ent_loss = entropy_loss_func(logit)
        with tape.stop_recording():
            r_vat = generate_vat_perturbation(model,embedded_ul,logit_ul,eta,num_iter)
        logit_vat = model(embedded_ul + r_vat,True)
        vat_loss = kld_loss_func(logit_ul,logit_vat)
        unsup_loss = ent_loss + vat_loss
        
        total_loss = sup_w*sup_loss + weight*unsup_loss

    grad = tape.gradient(total_loss,model.trainable_variables)
    gs = optimizer.apply_gradients(zip(grad,model.trainable_variables))
    return total_loss,logit,gs

def at_vat_step(model,optimizer,loss_func,batch,eta,weight=0.5,num_iter=2):
    sup_w = 1.0 - weight
    with tf.GradientTape() as tape:
        embedded = model.embedder(batch["lab"]["x"])
        logit = model(embedded,True)
        with tape.stop_recording():
            r_at = generate_at_perturbation(model,loss_func,embedded,batch["lab"]["y"],eta)
        logit_at = model(embedded+r_at,True)
        sup_loss = loss_func(batch["lab"]["y"],logit) + loss_func(batch["lab"]["y"],logit_at)
        
        embedded_ul = model.embedder(batch["unlab"]["x1"])
        logit_ul = model(embedded_ul,True)
        ent_loss = entropy_loss_func(logit)
        with tape.stop_recording():
            r_vat = generate_vat_perturbation(model,embedded_ul,logit_ul,eta,num_iter)
        logit_vat = model(embedded_ul + r_vat,True)
        vat_loss = kld_loss_func(logit_ul,logit_vat)
        unsup_loss = ent_loss + vat_loss
        
        total_loss = sup_w*sup_loss + weight*unsup_loss

    grad = tape.gradient(total_loss,model.trainable_variables)
    gs = optimizer.apply_gradients(zip(grad,model.trainable_variables))
    return total_loss,logit,gs

class customScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self,start,max,min,warmup,step_per_epoch):
        self.start = start
        self.max = max
        self.min = min
        self.warmup = warmup
        self.step_per_epoch = step_per_epoch
    
    def __call__(self,step):
        # __call__ method return learning rate at evert step
        # do sth w.r.t lr
        # return lr
        # if U want to adjust by epoch, do the following :
        # current_epoch = step//step_per_epoch -> do sth with current_epoch
        pass
        
        
