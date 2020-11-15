from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.metrics import roc_auc_score,confusion_matrix,balanced_accuracy_score,f1_score,precision_score,recall_score,matthews_corrcoef,make_scorer,classification_report
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE,VarianceThreshold
from sklearn.compose import ColumnTransformer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB,MultinomialNB,ComplementNB,BernoulliNB

import time,copy,pickle,os,time,functools
import numpy as np
import pandas as pd

from util.utils import *
from util.arg import *
from data_loader import GetData
from imblearn.over_sampling import SMOTE,RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from tqdm import tqdm
from rdkit.Chem import QED

import matplotlib.pyplot as plt
import seaborn as sns

data_path = args["data_path"]
train_path = args["train_path"]
test_path = args["test_path"]

def recall_warp(y_true,y_pred):
    return recall_score(y_true,y_pred,zero_division=0)

class train_model:
    """Sklearn GridSearch Class"""
    def __init__(self,data):
        self.data = data
        
    def GridSearch(self,model,args,fp_type,MD=False,normMD=False,balance=True,w=None,T=100,cv=5,**kwargs):
        """
        model : model is a str, representing sklearn model function
        args : A dictionary from util.utils, for Cross Validation for the model
        fp_type : Fingerprint Type, one of self.data.all_fps
        **kwargs : arguments for gridSearchCV(EX, cv, n_jobs, verbose)
        """
        assert fp_type in self.data.all_fps or fp_type is None

        if not MD:
            normMD = False
            
        scoring={"Accuracy":"accuracy" ,
                 "AUC":"roc_auc",
                 "Balanced_acc":"average_precision",
                 "F1":"f1",
                 "Presicion":"precision",
                 "Recall":"recall",
                 "MCC":make_scorer(matthews_corrcoef)}
        model_inst = globals()[str(model)]()
        print("\n***** Train {}_{}_{} *****\n".format(model,fp_type,MD))
        
        if fp_type is None and MD:
            inp = self.data.selected_MD
        elif fp_type is not None and MD:
            inp = self.data.concat(fp_type)
        else:
            inp = self.data.FP(fp_type)
        
        if normMD:
            #total_feat_num = self.data.concat(fp_type).shape[1]
            md_feat_num = self.data.selected_MD.shape[1]
            
            ct = ColumnTransformer([("md_norm",StandardScaler(),list(range(md_feat_num)))],remainder='passthrough')
            
            pipe = Pipeline([("ss",ct),("est",model_inst)])
            model_inst = pipe
            args = args_for_pipe(args)
            
        # refit set best_parameter by "refit" score
        spliter = StratifiedKFold(cv,random_state=1234,shuffle=True)
        model_grid = GridSearchCV(estimator=model_inst
                                       ,scoring=scoring,param_grid=args,cv=spliter,**kwargs)
        start= time.time()
        if balance:
            sample_weights = balanced_weights(self.data.label,w=w,T=T)
            
            if normMD:
                fit_args = {"est__"+"sample_weight":sample_weights}
            else:
                fit_args = {"sample_weight":sample_weights}
                
        else:
            fit_args = {}
                
        model_grid.fit(inp,self.data.label,**fit_args)
        end= time.time()
        print("\n{}, time consumed : {:.4f}\n".format(str(model),end-start))

        return model_grid
    
    def TrainModel(self,model,fp_type,balance=True,MD=False,normMD=False,w=None,T=100,**kwargs):
        assert fp_type in self.data.all_fps or fp_type is None
        model_instance = globals()[str(model)](**kwargs)

        print("\n***** Train {}_{}_{} *****\n".format(model,fp_type,MD))
        
        if fp_type is None and MD:
            inp = self.data.selected_MD
        elif fp_type is not None and MD:
            inp = self.data.concat(fp_type)
        else:
            inp = self.data.FP(fp_type)

        if normMD:
            pipe = Pipeline([("ss",StandardScaler()),("est",model_instance)])
            model_instance = pipe

        if balance:
            sample_weights = balanced_weights(self.data.label,w=w,T=T)
            
            if normMD:
                fit_args = {"est__"+"sample_weight":sample_weights}
            else:
                fit_args = {"sample_weight":sample_weights}                
        else:
            fit_args = {}
        model_instance.fit(inp,self.data.label,**fit_args)
        return model_instance
                       
    
    @staticmethod
    def scrutinize_model(model,X,y,out_path,out_path2):
        cv_result_table = get_table_result(model)
        print(cv_result_table)
        print()
        get_best,score_dict = get_confusion(model,X,y)
        print(get_best)
        cv_result_table.to_csv(out_path)
        get_best.to_csv(out_path2)
        with open(out_path2,"a") as f:
            for key,val in score_dict.items():
                f.write("{} :, {}\n".format(key,val))
            
    @property
    def get_all_model_list(self):
        filtered = filter(lambda x: "ECFP" in x or "MACCS" in x, dir(self))
        return list(filtered)

# seeds : 691 8748 4295 4980 2892 1577 8014  942 7245 4864
# seeds : 5246 2186 2223 8560  579 8434 2990 8121 8648 9089(MW)

##### Data Instance Load
seed = 5246
train_pkl = f"data_train_curated_{seed}_org.pkl" #f"data_train_curated_{seed}.pkl" # "data_train.pkl"
test_pkl = f"data_test_curated_{seed}_org.pkl"#f"data_test_curated_{seed}.pkl" # "data_test.pkl"
with open(os.path.join(data_path,train_pkl),"rb") as f:    
    data = pickle.load(f)

with open(os.path.join(data_path,test_pkl),"rb") as f:    
    data_test = pickle.load(f)

print(data.all_fps)
print(np.mean(data.label==1),np.mean(data_test.label==1))


##### Model build
model = train_model(data)

#### Model OverSampled
aug_ratio = .75
data_ros_aug = copy.deepcopy(data)
data_ros_aug.augment_RandomOver(sampling_strategy=aug_ratio,random_state=1234)
model_ros = train_model(data_ros_aug)

##### SMOTE
data_sm_aug = copy.deepcopy(data)
data_sm_aug.augment(aug_func=SMOTE,sampling_strategy=.75,k_neighbors=3,random_state=1234)
model_sm = train_model(data_sm_aug)

##### RUS
data_rus = copy.deepcopy(data)
data_rus.augment(aug_func=RandomUnderSampler,sampling_strategy=1.,random_state=1234)
model_rus = train_model(data_rus)


base_dir = "result"
def path_creator(sw,md,base):
    if sw:
        sw = "SampleWeights"
    else:
        sw = "NoSampleWeights"
    if md:
        md = "MolecularDescriptors"
    else:
        md = "FingerprintOnly"
    
    return_path = os.path.join(str(base),str(sw),str(md))
    if os.path.exists(return_path):
        return return_path
    else:
        os.makedirs(return_path,exist_ok=True)
        return return_path

def train_helper_make(model,aug_models,data_test,aug_types=["no"],cv=5,n_jobs=5,refit="Accuracy",verbose=1,fdir="abcd"):  
    
    def train_helper(model_names,balance,MD,normMD,w=None,T=100):
        args_dict = {name:args_map[name] for name in model_names}
        path = path_creator(balance,MD,fdir)
        for name in model_names:
        
            if MD:
                target_keys = model.data.all_fps
            else:
                target_keys = [None]+model.data.all_fps                
            
            for fp_key in target_keys:
                
                path1=os.path.join(path,f"{name}_table_{fp_key}.csv")
                path2=os.path.join(path,f"{name}_conf_{fp_key}.csv")
                             
                _gridcv = model.GridSearch(name,args_dict[name],
                    fp_key,MD,normMD,balance,w,T,cv=cv,n_jobs=n_jobs,verbose=verbose,refit=refit)
                
                if MD:
                    select_params = _gridcv.best_estimator_["est"].get_params()
                    test_set = data_test.concat
                else:
                    select_params = _gridcv.best_estimator_.get_params()
                    test_set = data_test.FP
                    
                if refit:
                    print("Refitting...")
                    model_refit = model.TrainModel(name,fp_key,balance=balance,MD=MD,normMD=normMD,w=w,T=T,**select_params)
                    get_best,score_dict = get_confusion(model_refit,test_set(fp_key),data_test.label)
                    cv_result_table = get_table_result(_gridcv)
                    print(get_best)
                    cv_result_table.to_csv(path1)
                    get_best.to_csv(path2)
                    with open(path2,"a") as f:
                        for key,val in score_dict.items():
                            f.write("{} :, {}\n".format(key,val))
                                            
                    if aug_models is not None:
                        for aug_type,aug_model in zip(aug_types,aug_models):
                            path3=os.path.join(path,f"{aug_type}_{name}_conf_{fp_key}.csv")
                            aug_model_refit = aug_model.TrainModel(name,fp_key,balance=balance,MD=MD,normMD=normMD,w=w,T=T,**select_params)
                            aug_get_best,aug_score_dict = get_confusion(aug_model_refit,test_set(fp_key),data_test.label)    
                            print(aug_get_best)
                            aug_get_best.to_csv(path3)
                            with open(path3,"a") as f:
                                for key,val in aug_score_dict.items():
                                    f.write("{} :, {}\n".format(key,val))
                            
                else:
                    model.scrutinize_model(_gridcv,test_set(fp_key),data_test.label,path1,path2)
                
    return train_helper

# Model to Augment fit #
all_model_names = [
    "RandomForestClassifier","SVC","LogisticRegression",
    "GradientBoostingClassifier","MLPClassifier","GaussianNB","KNeighborsClassifier"]
         
sw_model_names = ["LogisticRegression","RandomForestClassifier", "SVC"]

start = time.time()

model_train_helper = train_helper_make(model,[model_ros,model_sm],data_test=data_test,aug_types=["ros","sm"],cv=5,n_jobs=5,verbose=1,refit="F1",fdir=f"{base_dir}/no")

model_train_helper(all_model_names,balance=False,MD=False,normMD=True,w=None,T=1)
model_train_helper(all_model_names,balance=False,MD=True,normMD=True,w=None,T=1)
model_train_helper(sw_model_names,balance=True,MD=False,normMD=True,w=None,T=1)
model_train_helper(sw_model_names,balance=True,MD=True,normMD=True,w=None,T=1)
end = time.time()
print("Took {} secs".format(end - start))

# RUS fit #
start = time.time()

model_rus_train_helper = train_helper_make(model_rus,None,data_test=data_test,aug_types=[],cv=5,n_jobs=5,verbose=1,refit="F1",fdir=f"{base_dir}/rus")

model_rus_train_helper(all_model_names,balance=False,MD=False,normMD=True,w=None,T=1)
model_rus_train_helper(all_model_names,balance=False,MD=True,normMD=True,w=None,T=1)
model_rus_train_helper(sw_model_names,balance=True,MD=False,normMD=True,w=None,T=1)
model_rus_train_helper(sw_model_names,balance=True,MD=True,normMD=True,w=None,T=1)
end = time.time()
print("Took {} secs".format(end - start))
