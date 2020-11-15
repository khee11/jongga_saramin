# RDkit and modred
from rdkit import DataStructs 
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem # AllChem.GetMorganFIngerprint -> ECFP
from rdkit.Chem import MACCSkeys # MACCS fingerprint
#from rdkit.Chem import Descriptors
from rdkit.Chem.AtomPairs import Pairs,Torsions # Pairs.GetAtomPairFingerprint, Torsions.GetTopologicalTorsionFingerprint
from rdkit.Chem.AllChem import PatternFingerprint,RDKFingerprint,LayeredFingerprint
from rdkit.Chem.MolStandardize.rdMolStandardize import Normalize,StandardizeSmiles

from mordred import Calculator, descriptors

from sklearn.feature_selection import RFE,VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE,RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler,MinMaxScaler
# sklearns and keras
import pandas as pd
import numpy as np
import pickle,os,copy,time,random

from util.utils import *

"""
Fingerprint list
1. AllChem.GetMorganFingerprintAsBitVect
2. MACCSkeys.GenMACCSKeys
3. AllChem.RDKFingerprint
4. AllChem.PatternFingerprint
5. AllChem.LayeredFingerprint
"""
Name = "Substance Name"
Number = "Substance Number"
Type= "Number type"

seed = 5246
train_path =  f"dataset/TG471_train_all_stdn_curated_mw_{seed}_org.csv" 
test_path = f"dataset/TG471_test_all_stdn_curated_mw_{seed}_org.csv"
smile_col = "res_stnd_SMILES"
label_col = "Result"

train_pkl = f"data_train_curated_{seed}_org.pkl"
test_pkl = f"data_test_curated_{seed}_org.pkl"
md_list_file = "selected_md.txt"

result_path = os.path.join("data")
if not os.path.exists(result_path):
    os.mkdir(result_path)

sampling= "ros"
ratio=0.5

mol_norm = False
smi_stnd = False

fp_funcs_args = {
    "ECFP":(AllChem.GetMorganFingerprintAsBitVect,{"radius":2,"nBits":1024}),
    "MACCS":(MACCSkeys.GenMACCSKeys,{}),
    "RDK":(AllChem.RDKFingerprint,{"minPath":1,"maxPath":7,"fpSize":2048,"useHs":True}),
    "PATT":(AllChem.PatternFingerprint,{}),
    "LAY":(AllChem.LayeredFingerprint,{}),
    #"PubChem":(Pubchem_FP,{"fingerprints":True,"descriptors":False})
    }

class GetData:
    
    def __init__(self,data_path,config,is_excel=True):
        self.data_path = data_path
        self.is_excel=is_excel # data format excel? or csv?
        self._fp_dict = {} # enroll whole fp
        self.config =config
                
    def load_file(self,label_col,smile_col,skiprows=2,sheet_name=0):
        self.label_col = label_col
        self.smile_col = smile_col
        if self.is_excel:
            self.raw_data = pd.read_excel(self.data_path,skiprows=skiprows,sheet_name=sheet_name)
        else:
            self.raw_data = pd.read_csv(self.data_path)
        self._error_mask = [False]*len(self.raw_data) # error case throughout whole process(True denotes the errored)
        print("\nSMILES columns : {}".format(self.smile_col))
        print("\nTotal Rows : {}\n".format(len(self.raw_data)))
        
    def label_encode(self):
        encoder = LabelEncoder()
        self._label=encoder.fit_transform(self.raw_data[self.label_col])
        print("\nTotal Classes : {}\n".format(len(np.unique(self._label))))
    
    def get_mols(self,normalize=False,smi_stnd=False):
        mol_lst = []
        smiles_raw = self.raw_data[self.smile_col]
        for i in range(len(self.raw_data)):
            smile = smiles_raw[i]
            try:
                mol=Chem.MolFromSmiles(smile)
                if normalize:
                    mol = Normalize(mol)
                mol_lst.append(mol)
                if mol is None:
                    self._error_mask[i] = True
            except Exception as e:
                print(e)
                mol_lst.append(None)
                self._error_mask[i] = True
        self._mol_lst = mol_lst
        
    def get_fps(self,fp_type,fp_func,**kwargs):
        fps = []
        for i,mol in enumerate(self._mol_lst):
            try:
                fp=fp_func(mol,**kwargs)
                arr = np.zeros((1,))
                DataStructs.ConvertToNumpyArray(fp, arr)
                fps.append(list(arr))
            except Exception as e: # 보통은, smile이 존재하면 error는 안뜨는듯
                print(e)
                fps.append(None)
                self._error_mask[i]= True

        self._fp_dict.update({fp_type:fps}) # register whole fp

    def get_MD(self,ignore_3D=True):
        """
        Get MD ONLY for non-error cases
        """
        calc = Calculator(descriptors,ignore_3D=ignore_3D)
        error_cases = np.squeeze(np.argwhere(self._error_mask))
        mol_noError = list_where(self._mol_lst,error_cases,False) # index(error_cases)에 없으면 가져옴
        mol_descriptor = calc.pandas(mol_noError)
        self._MD = mol_descriptor.astype("float64")
    
    def feature_select(self,rfe_step=5,max_to_select=None,sampling="ros",ratio=0.5):
        self._selected_MD = None
        
        # Take only non error cases(MD and corresponding labels)
        _MD = copy.deepcopy(self._MD)
        _label = self.get_no_error("label",True)#copy.deepcopy(self._label)[np.logical_not(self._error_mask)]
        
        _selected_mds = []
        
        _MD = _MD.dropna(axis=1)
        cols =_MD.columns
        
        print("Selecting Molecular descriptors...")
        for step in range(10):
            print("Step {}".format(step))
            mm_ss = MinMaxScaler()
            mm_ss.fit(_MD)
            
            MD = mm_ss.transform(_MD)
            MD = pd.DataFrame(MD,columns=cols)
            
            var_threshold = self.config["var_threshold"] #0.05
            VarianceSelector=VarianceThreshold(threshold=var_threshold)
            VarianceSelector.fit(MD)
            included_step1= MD.columns[VarianceSelector.get_support()]

            MD = MD[included_step1]
        
            MD = MD.replace([-np.inf, np.inf],np.nan).dropna(axis=1)
            MD = MD.fillna(MD.mean()) # or 0.?
      
            if sampling=="ros":
                sampler=RandomOverSampler(sampling_strategy=ratio)
                MD,label=sampler.fit_resample(MD , _label)
            elif sampling=="rus":
                sampler=RandomUnderSampler(sampling_strategy=ratio)
                MD,label=sampler.fit_resample(MD , _label)
            
            MD = pd.DataFrame(MD,columns=MD.columns)
            
            _dict = {}
            for col in MD:
                try:
                    _dict[col]=pearson_corr(MD[col].values,label)
                except:
                    _dict[col]=0.0 # error removed
                
            MD_corr = MD.corr().abs()    
            corr_threshold = self.config["corr_threshold"]
            correlated =np.argwhere(np.triu(MD_corr>corr_threshold,k=1))

            all_set = set(MD_corr.columns)
            excluded_set = set()
            for i,j in correlated:
                cand1=MD.columns[i] ; cand2=MD.columns[j] 
                val1 = _dict[cand1] ; val2 = _dict[cand2] 
                if val1>=val2:
                    excluded_set.add(cand2)
                else :
                    excluded_set.add(cand1)

            included_step2 = all_set - excluded_set
            MD = MD[included_step2]

            _TREE=RandomForestClassifier(
                class_weight="balanced",max_depth=5,min_samples_split=0.01,max_features="auto")
            _TREE.fit(MD.values, label)#,sample_weight=balanced_weights(label,T=1))
        
            imp_threshold = self.config["imp_threshold"]
            included_step3=MD.columns[_TREE.feature_importances_ > imp_threshold]
            MD= MD[included_step3]
        
            if max_to_select is None:
                max_to_select = int(len(included_step3)/2)
            param_grid = {"n_features_to_select":np.arange(10,max_to_select,rfe_step)}
            scoring = {"ACC":"accuracy","AUC":"roc_auc"}
            rfe=RFE(RandomForestClassifier(
                class_weight="balanced",max_depth=5,min_samples_split=0.01,max_features="auto"),step=0.05)
            rfe_grid = GridSearchCV(rfe,param_grid=param_grid,n_jobs=6,cv=2,verbose=1,scoring=scoring,
                refit="AUC")
            rfe_grid.fit(MD.values,label)
                      
            included_step4=rfe_grid.best_estimator_.support_
    
            MD = MD.iloc[:,included_step4]
            
            _selected_mds.append(set(MD.columns))
        
        self._final_columns = sorted(list(set.intersection(*_selected_mds)))
        self._selected_MD = self._MD[self._final_columns]
        print("Total {} MD selected".format(len(self._final_columns)))
        with open(md_list_file,"w") as f:
            for col in self._final_columns:
                f.write(col+"\n")
        
    def finalize(self):
        """
        Remove all error cases, from any step(MD,FP,feature selection etc)
        """
        error_cases = np.squeeze(np.argwhere(self._error_mask)).astype(np.int32) # 전체 데이터중 error인 index
        
        if len(error_cases)==0:
            for fp_name,fp_obj in self._fp_dict.items():
                fp_obj = np.array(fp_obj)
                self._fp_dict.update({fp_name:fp_obj}) # Ensure replacement
            print("All cases successfully converted")    
            return True
        
        for key in self._fp_dict.keys():
            _lst = self._fp_dict[key]
            lst = list_where(_lst,error_cases) # index(error_cases)에 없으면 가져옴
            _lst = np.array(lst)
            self._fp_dict.update({key:_lst})
            
        self._label = np.delete(self._label,error_cases,axis=0)
    
    @property
    def selected_MD(self):
        return self._selected_MD
    @property
    def final_columns(self):
        return self._final_columns
    @property
    def MD(self):
        return self._MD
    
    @property
    def label(self):
        return self._label
    
    @property
    def all_fps(self):
        return list(self._fp_dict.keys())
    
    @property
    def mol_lst(self):
        return self._mol_lst
    
    def FP(self,fp):
        if fp is None:
            return self._selected_MD
        assert fp in self.all_fps,"{} not in data".format(fp)
        return self._fp_dict[fp]
    
    def MD_from(self,md_list,is_np=False):
        if is_np:
            self.MD[md_list].value
        else:
            self.MD[md_list]
    
    def concat(self,fp_type):
        """Concat FP and MD"""
        if fp_type is None:
            return self._selected_MD
        selected_MD = copy.deepcopy(self.selected_MD)
        lst = [selected_MD,self.FP(fp_type)]
        return np.concatenate(lst,axis=1).astype(np.float32)
    
    def under_RandomOver(self,**kwargs):
        if "random_state" not in kwargs:
            kwargs.update({"random_state":random.randint(0,1e6)})

        rus = RandomUnderSampler(**kwargs)
            
        if hasattr(self,"_selected_MD"):
            self._selected_MD,label = rus.fit_resample(self._selected_MD,self._label)

        if hasattr(self,"_fp_dict"):
            _dict = {}
            for key in self._fp_dict.keys():
                arr = self._fp_dict[key]
                arr,lbl = rus.fit_resample(arr,self._label)
                _dict[key] = lbl
                self._fp_dict.update({key:arr})
        
        try:
            self._label = label
        except:
            self._label = lbl
    
    def augment_RandomOver(self,**kwargs):
        
        if "random_state" not in kwargs:
            kwargs.update({"random_state":random.randint(0,1e6)})

        ros = RandomOverSampler(**kwargs)
            
        if hasattr(self,"_selected_MD"):
            self._selected_MD,label = ros.fit_resample(self._selected_MD,self._label)

        if hasattr(self,"_fp_dict"):
            _dict = {}
            for key in self._fp_dict.keys():
                arr = self._fp_dict[key]
                arr,lbl = ros.fit_resample(arr,self._label)
                _dict[key] = lbl
                self._fp_dict.update({key:arr})
               
        try:
            self._label = label
        except:
            self._label = lbl
    
    def augment(self,aug_func,**kwargs):
        if "random_state" not in kwargs:
            kwargs.update({"random_state":random.randint(0,1e6)})

        sampler = aug_func(**kwargs)
            
        if hasattr(self,"_selected_MD"):
            self._selected_MD,label = sampler.fit_resample(self._selected_MD,self._label)

        if hasattr(self,"_fp_dict"):
            _dict = {}
            for key in self._fp_dict.keys():
                arr = self._fp_dict[key]
                arr,lbl = sampler.fit_resample(arr,self._label)
                _dict[key] = lbl
                self._fp_dict.update({key:arr})
               
        try:
            self._label = label
        except:
            self._label = lbl   

    def get_no_error(self,attr,is_copy=False):
        if is_copy:
            target = copy.deepcopy(getattr(self,attr))
        else:
            target = getattr(self,attr)
        return target[np.logical_not(self._error_mask)]
        
# dictionary D : idx,name,smiles,mol,fp_vec,...,fp_vec_n,MD,selected_MD,is_error
# re-build D s.t., for key,val in D.items(): if
def create_train():
    from util.arg import args_data
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    else:
        print("Output will be printed at {}".format(result_path))
        
    data = GetData(train_path,args_data,False)
    print("Load raw file...")
    data.load_file(label_col,smile_col,skiprows=2,sheet_name=0)
    print("Labeling...")
    data.label_encode()
    print("Get moleculars...")
    data.get_mols(mol_norm) # 총 62개가 None
    print("Get moleculars fingerprints...")
    for key,(func,args) in fp_funcs_args.items():
        print("Fingerprint : {}...".format(key))
        data.get_fps(key,func,**args)
    print("Get molecular descriptors...")
    data.get_MD() #
    data.feature_select(10,None,sampling=sampling,ratio=ratio) #
    data.finalize()
    #data.to_csv(os.path.join(result_path,"train_data"),"_"+suffix) #
    with open(os.path.join(result_path,train_pkl),"wb") as f:
        pickle.dump(data,f)

def create_test():
    selected_cols=read_lines("selected_md.txt")
     
    data_test = GetData(test_path,args_data,False) # Or can be diffrent path
    print("Load raw file...")
    data_test.load_file(label_col,smile_col,skiprows=1,sheet_name=1)
    print("Labeling...")
    data_test.label_encode()
    print("Get moleculars...")
    data_test.get_mols(mol_norm)
    print("Get moleculars fingerprints...")
    for key,(func,args) in fp_funcs_args.items():
        print("Fingerprint : {}...".format(key))
        data_test.get_fps(key,func,**args)
    print("Get molecular descriptors...")
    data_test.get_MD() #
    data_test._final_columns = pd.Index(selected_cols) # 
    data_test._selected_MD = data_test.MD[selected_cols] #
    data_test.finalize()
    with open(os.path.join(result_path,test_pkl),"wb") as f:
        pickle.dump(data_test,f)

if __name__=="__main__":
    from util.arg import args_data
    
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    else:
        print("Output will be printed at {}".format(result_path))
    
    mode = "train" # train , test
    if mode=="train":
        create_train()
    elif mode =="test":
        create_test()

