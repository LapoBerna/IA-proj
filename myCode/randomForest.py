import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataRefining import splitData
from dataRefining import getRefinedStudents
from getFeatureImportances import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report 
from sklearn.inspection import permutation_importance
from dataclasses import dataclass 

@dataclass
class trainedForest:
    name: str
    forest: RandomForestClassifier
    X_test: pd.DataFrame
    prediction: pd.DataFrame
    y_test: pd.DataFrame

def rfClassifier(name, seed, X_train, y_train, X_test, y_test):
    
    rfc = RandomForestClassifier(random_state=seed)
    rfc.fit(X_train, y_train)
    FS_TMA1_pred = rfc.predict(X_test)
    return trainedForest(name, rfc, X_test, FS_TMA1_pred, y_test)

def randomForestClassification(trainDim, seed, codeModule, codePresentation, mode):
    X_train, X_test, FS_TMA1_train, FS_TMA1_test, FS_TMA2_train, FS_TMA2_test, FS_TMA3_train, FS_TMA3_test, FS_TMA4_train, FS_TMA4_test, FS_TMA5_train, FS_TMA5_test, FS_TMAF_train, FS_TMAF_test = splitData(trainDim, seed, codeModule, codePresentation, mode)

    forest_TMA1 = rfClassifier("RF_TMA1", seed, X_train, FS_TMA1_train, X_test, FS_TMA1_test)
    forest_TMA2 = rfClassifier("RF_TMA2", seed, X_train, FS_TMA2_train, X_test, FS_TMA2_test)
    forest_TMA3 = rfClassifier("RF_TMA3", seed, X_train, FS_TMA3_train, X_test, FS_TMA3_test)
    forest_TMA4 = rfClassifier("RF_TMA4", seed, X_train, FS_TMA4_train, X_test, FS_TMA4_test)
    forest_TMA5 = rfClassifier("RF_TMA5", seed, X_train, FS_TMA5_train, X_test, FS_TMA5_test)
    forest_TMAF = rfClassifier("RF_TMAF", seed, X_train, FS_TMAF_train, X_test, FS_TMAF_test)

    
    return (forest_TMA1, forest_TMA2, forest_TMA3, forest_TMA4, forest_TMA5, forest_TMAF)
 
def makeStats(forestStruct, seed, mode):
    
    feat_importances = getFeatureImportances(forestStruct.forest.feature_names_in_, forestStruct.forest.feature_importances_)
    
    meanImpurityDecrease(forestStruct, mode)
    featurePermutation(forestStruct, seed, mode)

    with open("stats and plots/Random Forests/"+ mode + "/" + mode + "_" + forestStruct.name + "_textfile.txt","w+") as f:
        f.write(np.array2string(feat_importances.index.values, separator=', ', max_line_width=150))
        f.write("\n")
        f.write(np.array2string(feat_importances.values*100, separator=', ', max_line_width=150))###
        f.write("\n")
        f.write(np.array2string(confusion_matrix(forestStruct.y_test, forestStruct.prediction), separator=', ', max_line_width=150))
        f.write("\n")
        f.write(classification_report(forestStruct.y_test, forestStruct.prediction, zero_division=0))
        f.close()
         
def makeBench(forestStruct, X_bench, y_bench):
    
    X_pred = forestStruct.forest.predict(X_bench)
    print(classification_report(X_pred, y_bench, zero_division=0))  

def meanImpurityDecrease(forestStruct, mode):

    forest_importances = getFeatureImportances(forestStruct.forest.feature_names_in_, forestStruct.forest.feature_importances_)
    std = np.std([getFeatureImportances(forestStruct.forest.feature_names_in_, tree.feature_importances_) for tree in forestStruct.forest.estimators_], axis=0)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.savefig("stats and plots/Random Forests/"+ mode + "/" + mode + "_" + forestStruct.name +'_mean_Impurity_Decrease.svg', format='svg')

def featurePermutation(forestStruct, seed, mode):

    result = permutation_importance(forestStruct.forest, forestStruct.X_test, forestStruct.y_test, n_repeats=100, random_state=seed, n_jobs=4)
    
    sorted_importances_idx = result.importances_mean.argsort()
    importances = pd.DataFrame(
        result.importances[sorted_importances_idx].T,
        columns=forestStruct.X_test.columns[sorted_importances_idx],
    )
    ax = importances.plot.box(vert=False, whis=10)
    ax.set_title("Permutation Importances (test set)")
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_xlabel("Decrease in accuracy score")
    ax.figure.tight_layout()
    plt.savefig("stats and plots/Random Forests/"+ mode + "/" + mode + "_" + forestStruct.name +'_feature_Permutation.svg', format='svg')

