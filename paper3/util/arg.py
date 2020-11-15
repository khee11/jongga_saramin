import os
from sklearn.tree import DecisionTreeClassifier as DT
result_path = "result"

args = {
    "n_jobs":4,
    "cv":10,
    "data_path":os.path.join("data"),
    "train_path":os.path.join(result_path,"train"),
    "test_path":os.path.join(result_path,"test"),
}

args_data = {
    "radius" : 2,
    "nbits" : 1024,
    "var_threshold" : 1e-3,
    "corr_threshold" : 0.90,
    "imp_threshold" : 1e-3
}

args_gbt = {
    "min_samples_leaf":[0.005,0.01],"max_depth":[3,5,7,9],"max_features":["auto"]
}
args_svm = {
    "C" : [0.75,1.0,1.25,1.5],"class_weight":[None],"gamma":["scale","auto"] # balanced제거
}
args_rf = {
    "n_estimators" : [500],"max_depth" : [3,5,7,9],"min_samples_split" : [0.01,0.005],"class_weight": [None] # balanced제거
}
args_knn = {
    "n_neighbors" : [3,5,7,9,11],"metric" : ["manhattan","minkowski"],"weights" : ["distance","uniform"]
}
args_mlp = {
    "hidden_layer_sizes":[(50,),(30,),(10,)],"activation": ["tanh","relu"],"solver":["lbfgs"],"alpha":[1e-1,1e-2],"max_iter":[1000]
}
args_lgr={
    "penalty":["l1"],"C":[0.75,1.0,1.25,1.5],"fit_intercept":[False],"class_weight":[None],"max_iter":[1000],"solver":['liblinear'] # balanced제거
}
args_nb={}

args_map = {
    "LogisticRegression":args_lgr,
    "RandomForestClassifier":args_rf,
    "GaussianNB":args_nb,
    "BernoulliNB":args_nb,
    "MLPClassifier":args_mlp,
    "GradientBoostingClassifier":args_gbt,
    "KNeighborsClassifier":args_knn,
    "SVC":args_svm
}


