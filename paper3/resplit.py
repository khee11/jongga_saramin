import pandas as pd
import numpy as np
import random,os
from rdkit.Chem.MolStandardize.rdMolStandardize import Normalize,StandardizeSmiles
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Chem.Draw import MolToImage
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.Descriptors import MolWt,ExactMolWt

data_path = "dataset"
split_ratio = 0.8
    
def split_df(df,seed=1234):
    total_num = len(df)
    train_num = int(total_num*split_ratio)
    df=df.sample(frac=1.0,random_state=seed).reset_index(drop=True)
    TG471_train = df[:train_num]
    TG471_test = df[train_num:]
    print(
        np.mean(TG471_train["Result"]=="P"),
        np.mean(TG471_test["Result"]=="P")
    )
    return TG471_train,TG471_test

df = pd.read_excel(f"{data_path}/TG471_data(curated).xlsx")
df = df.reset_index(drop=True)

"""
1. 중복 CAS ID 제거
2. 중복 SMILES 제거(결과 상충시 모두 제거, 아니면 아무거나 남겨둠)
3. SMILES 정규화
4. MOL로 바꾼 후에, SALT 제거
5. MW 너무 크거나(800) 작은거(40) 없앰
"""

# 1. 중복 CAS ID 제거
nrow,ncols = df.shape
if len(np.unique(df["Substance Number"]))==nrow:
    pass

# 2. 중복 SMILES 제거
unique_smi, smi_counts = np.unique(df["SMILES"],return_counts=True)
dup_index=np.squeeze(np.argwhere(smi_counts>1)) # 먼저 중복&상충 먼저 지우자

f=open("abcd","w",encoding="utf-8")
for i in dup_index:
    smi = unique_smi[i]
    print(df[df["SMILES"]==smi],file=f)
    print("\n",file=f)

dup_and_contra_list = []
for i in dup_index:
    smi = unique_smi[i]
    results = df[df["SMILES"]==smi]["Result"].values
    if np.all(results==results[0]):
        pass
    else:
        dup_and_contra_list.append(i)

for i in dup_and_contra_list: # check and remove all duplicates_and_contradict cases
    print(df[df["SMILES"]==unique_smi[i]])
    df = df[df["SMILES"]!=unique_smi[i]]
    print()

df = df.drop_duplicates("SMILES") # Leave only one case : Finally, All duplicates are removed
df=df.reset_index(drop=True)

# 3. SMILES 정규화 : 정규화 안되는 건 날려버림 ( salt 제거 먼저 할지 고려해봐야할듯)
def stnd_func(x):
    try:
        smi=StandardizeSmiles(x)
    except:
        smi="-"
    return smi
    
stnd_smis=df["SMILES"].apply(stnd_func)
df["STND_SMILES"] = stnd_smis

base_dir = "pic_raw_vs_stnd"
for i in range(len(df)):
    smi,stnd_smi = df.loc[i][["SMILES","STND_SMILES"]]
    current_dir= os.path.join(base_dir,str(i))
    os.makedirs(current_dir,exist=True)
    
    try:
        MolToImage(MolFromSmiles(smi)).save(os.path.join(current_dir,"smi.jpeg"))
    except Exception as e:
        print(i,e)
    
    try:
        MolToImage(MolFromSmiles(stnd_smi)).save(os.path.join(current_dir,"stnd_smi.jpeg"))
    except Exception as e:
        print(i,e)

assert len(set(df["STND_SMILES"].values))==len(df),"Standardize Smiles reduced"

non_cases = np.squeeze(np.argwhere((df["STND_SMILES"]=="-").values))

df = df.drop(non_cases.tolist())
df=df.reset_index(drop=True)

print(df[df.duplicated("STND_SMILES",False)][["Result","STND_SMILES"]]) # No Contradict

df = df.drop_duplicates("STND_SMILES") # Drop first duplicated ones
df=df.reset_index(drop=True)

# 4. Salt 제거
df = df.reset_index(drop=True)
remover = SaltRemover()
base_dir = "pic_salt"
res_smi_list = []
res_stnd_smi_list = []
for i in range(len(df)):
    smi,stnd_smi = df.loc[i][["SMILES","STND_SMILES"]]
    
    current_dir= os.path.join(base_dir,str(i))
    #os.mkdir(current_dir)

    try:
        mol = MolFromSmiles(smi)
        #MolToImage(mol).save(os.path.join(current_dir,"smi.jpeg"))
    except Exception as e:
        print(i,e)
    
    try:
        stnd_mol = MolFromSmiles(stnd_smi)
        #MolToImage(stnd_mol).save(os.path.join(current_dir,"stnd_smi.jpeg"))
    except Exception as e:
        print(i,e)

    try:
        res= remover.StripMol(mol)
        #MolToImage(res).save(os.path.join(current_dir,"smi_res.jpeg"))
        res_smi = MolToSmiles(res)
        res_smi_list.append(res_smi)
    except Exception as e:
        res_smi = "-"
        res_smi_list.append(res_smi)
        print(i,e)
    
    try:
        stnd_res = remover.StripMol(stnd_mol)
        #MolToImage(stnd_res).save(os.path.join(current_dir,"stnd_smi_res.jpeg"))
        res_stnd_smi = MolToSmiles(stnd_res)
        res_stnd_smi_list.append(res_stnd_smi)
    except Exception as e:
        res_stnd_smi = "-"
        res_stnd_smi_list.append(res_stnd_smi)
        print(i,e)

df["res_SMILES"] =res_smi_list # Salt Removed from SMILES
df["res_stnd_SMILES"] = res_stnd_smi_list # Salt Removed from Standardized SMILES

error_mask2 = np.any([df["res_stnd_SMILES"]=="",df["res_stnd_SMILES"]=="-" ],axis=0)

df3 = df[np.logical_not(error_mask2)]
print(df3[df3.duplicated("res_SMILES",False)][["Result","res_SMILES"]]) # CSV로 하나씩 살피기
df3 = df.drop_duplicates("res_stnd_SMILES") # Drop first duplicated ones

# 5. MW 짜르기
mw_lst = []
for i,row in df3.iterrows():
    smi=row["res_stnd_SMILES"]
    mol=MolFromSmiles(smi)
    mw= MolWt(mol)
    mw_lst.append(mw)
df3["MW"]=mw_lst
print(
    np.mean(df3[df3["MW"]<40]["Result"]=="P"),
    np.mean(df3[df3["MW"]>800]["Result"]=="P"))
df3_mw=df3[df3["MW"]>40]
df3_mw=df3_mw[df3_mw["MW"]<800]
df3_mw = df3_mw.reset_index(drop=True)

seeds= [1043]#np.random.randint(1,1e4,10)

for seed in seeds:
    trn,test = split_df(df3_mw,seed) # df3,df3_mw
    trn.to_csv(f"{data_path}/TG471_train_all_stdn_curated_mw_{seed}.csv")
    test.to_csv(f"{data_path}/TG471_test_all_stdn_curated_mw_{seed}.csv")

