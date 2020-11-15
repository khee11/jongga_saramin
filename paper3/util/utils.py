from rdkit.Chem import Draw
from os.path import join
from sklearn.metrics import roc_auc_score,confusion_matrix,balanced_accuracy_score,f1_score,precision_score,recall_score,matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from treeinterpreter import treeinterpreter as ti
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cirpy
from padelpy import from_smiles
from rdkit.Chem import MolToSmiles

# Parser helper
def parser_section(inp,type=str):
    def map_type(x):
        if type(x)==str and x =="None":
            return None
        else:
            return type(x)
        
    return list(map(map_type(x),inp.split(",")))

def read_lines(path):
    """Only read lines into a list"""
    return_lst = []
    with open(path,"r") as f:
        for line in f:
            return_lst.append(line.strip())
    return return_lst
    
def parse_config(config,types):
    keys = list(config)
    if type(types)==type:
        types = [types]*len(keys)
    _dict= {}
    for key,_type in zip(keys,types):
        x = parser_section(config[key],_type)
        _dict.update({key:x})
    return _dict

def list_where(lst,idx,is_remove=False):
    """is_remove : is idx for removing index?"""
    _len = len(lst)
    if is_remove:
        return [lst[i] for i in range(_len) if i in idx]
    else:
        return [lst[i] for i in range(_len) if i not in idx]

# feature selector helper
def pearson_corr(x,y,is_abs=True):
    n = len(x)
    assert n == len(y)
   
    nomin =np.sum((x-np.mean(x))*(y-np.mean(y)))/(n-1)
    denum = np.sqrt(np.var(x-np.mean(x))*np.var(y-np.mean(y)))
    return np.abs(nomin/denum)

def get_corr(x,y):
    logistic = LogisticRegression().fit(x,y)
    return logistic.score(x,y)

def get_table_result(grid_model):
    result = grid_model.cv_results_
    num_params = len(result["params"])
    print("number of parameter sets : {}".format(num_params))
    
    result_dict = {
        'Accuracy': result['mean_test_Accuracy'],
        'AUC':result['mean_test_AUC'],
        'F1': result['mean_test_F1'],
        'Presicion': result['mean_test_Presicion'],
        'Recall': result['mean_test_Recall'],
        "MCC":result['mean_test_MCC'],
        "param": result["params"]}
    
    result=pd.DataFrame(result_dict).sort_values(by=["Recall","Accuracy"],ascending=False)
    return result

def get_confusion(grid_model,X,y):
    y_pred = grid_model.predict(X)
    
    score_dict={}
    score_dict["Acc"] = np.mean(y_pred==y)
    score_dict["auc"] = roc_auc_score(y,y_pred)
    score_dict["f1"] = f1_score(y,y_pred)
    score_dict["precision"] = precision_score(y,y_pred)
    score_dict["recall"] = recall_score(y,y_pred)
    score_dict["mcc"] = matthews_corrcoef(y,y_pred)
    confusion=confusion_matrix(y,y_pred,labels=[1,0])
    
    print("Accuracy  : {:.4f}".format(score_dict["Acc"]))
    print("AUC : {:.4f}".format(score_dict["auc"]))
    print("Precision : {:.4f}".format(score_dict["precision"]))
    print("Recall : {:.4f}".format(score_dict["recall"]))
    print("MCC : {:.4f}".format(score_dict["mcc"]))
    print("F1 : {:.4f}".format(score_dict["f1"]))
    out = pd.DataFrame(confusion , index=[["True_y","True_y"],["positive","negative"]],
            columns=[["pred_y","pred_y"],["positive","negative"]])
    return out,score_dict

def cas_to_smile(cas_id):
    cas_id = str(cas_id).strip()
    smile=cirpy.resolve(cas_id,"smiles")
    if smile is None:
        return ''
    else:
        return smile

def softmax(x,T=100):
    denu = np.sum(np.exp(x/T))
    return np.exp(x/T)/denu

def balanced_weights(label,w=None,T=100):
    unique_label,label_count=np.unique(label, return_counts=True)
   
    if np.any(label_count==0) or len(label_count)==1:
        return None

    inv_count_ratio = (np.sum(label_count)-label_count)/np.sum(label_count)
    
    if w is None:
        label_weights = softmax(inv_count_ratio,T=T)
        sample_weights = np.zeros_like(label).astype(np.float32)
        for lbl,wght in zip(unique_label,label_weights):
            print("Label : {}, Weight :{:.4f}".format(lbl,wght))
            idx = np.squeeze(np.argwhere(label==lbl))
            sample_weights[idx] = wght
        return sample_weights
    else:
        w1 = 1-w
        w2 = w
        label_weights = [w1,w2]
        sample_weights = np.zeros_like(label).astype(np.float32)
        for lbl,wght in zip(unique_label,label_weights):
            print("Label : {}, Weight :{:.4f}".format(lbl,wght))
            idx = np.squeeze(np.argwhere(label==lbl))
            sample_weights[idx] = wght
        return sample_weights

def args_for_pipe(args):
    args_new = {}
    for key,val in args.items():
        args_new.update({"est__"+key:val})
    return args_new
    
def tree_interpret(model,X_test,cols=None):
    pred,bias,contrib=ti.predict(model,X_test)
    assert np.allclose(pred,np.sum(contrib,axis=1)+bias[0]),"something wrong!!!"
    if cols is None:
        try:
            cols = X_test.columns()
        except:
            cols = np.arange(X_test.shape[1]).astype(np.str)

    _contrib = pd.DataFrame(contrib[:,:,1],columns=cols)
    _pred = np.argmax(pred,axis=1)
    return _pred,bias,_contrib

def _get_color(value):
    """To make positive DFCs plot green, negative DFCs plot red."""
    green, red = sns.color_palette()[2:4]
    if value >= 0: return green
    return red

def indiv_dfc(contrib,pred,label,idx,K_TOP=10,save_path=None,subs_name=""):
    example = contrib.loc[idx]
    sort_idx = example.abs().sort_values(ascending=False)[:K_TOP].index # Take most K_TOP influential
    example = example[sort_idx]
    colors = example.map(_get_color)
    ax = example.to_frame().plot(kind="barh",color=[colors],legend=None,
                          alpha=0.75,figsize=(11,5))
    ax.set_title('Feature contributions for example {}\n{}\npred: {}; label: {}'.format(idx,subs_name, pred[idx],label[idx]),fontsize=9)
    ax.set_xlabel('Contribution to predicted probability', size=14)
    fig = ax.figure
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
    plt.close(fig)
    # fig.clf()

def Mol_add_atomSymbol(mol):
    for atom in mol.GetAtoms():
        atom.SetProp("atomLabel",atom.GetSymbol())

def Mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())

def Draw_from_svg(svg,path):
    with open(path,"w") as f:
        f.write(svg)

# add  symbol, organize path, then done
def draw_fp_bit(fp_func,base_path,mol,idx=None,**kwargs):
    info = {}
    Mol_add_atomSymbol(mol)
    fp = fp_func(mol,bitInfo=info,**kwargs)
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    if idx is None:
        for key in info.keys():
            print("Bit {} : \n".format(key))
            sub_svg=Draw.DrawMorganBit(mol,key,info,useSVG=True)
            key = str(key)
            full_path  = join(base_path,key)
            Draw_from_svg(sub_svg,full_path)      
    else:
        sub_svg=Draw.DrawMorganBit(mol,idx,info,useSVG=True)
        Draw_from_svg(sub_svg,join(base_path,idx))

def Fp2arr(fp):
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp,arr)
    return arr
    
def Fp_with_bi(func,mol,**kwargs):
    info = {}
    fp = func(mol,bitInfo=info,**kwargs)
    return fp,info
    
def Pubchem_FP(smi,**kwargs):
    if type(smi) != str:
        smi = MolToSmiles(smi)
    fp = from_smiles(smi,**kwargs)
    lst_as_vect = []
    for key,val in fp.items():
        lst_as_vect.append(val)
    assert len(lst_as_vect)==881
    return lst_as_vect
