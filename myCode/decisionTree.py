import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataRefining import splitData
from dataRefining import getRefinedStudents
from getFeatureImportances import *
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report 
from dataclasses import dataclass 

@dataclass
class trainedTree:
    name: str
    tree: DecisionTreeClassifier
    prediction: pd.DataFrame
    test: pd.DataFrame

def dtClassifier(name, seed, X_train, y_train, X_test, y_test):

    dtc = DecisionTreeClassifier(random_state=seed)
    dtc.fit(X_train, y_train)
    FS_TMA1_pred = dtc.predict(X_test)
    return trainedTree(name, dtc, FS_TMA1_pred, y_test)

def decisionTreeClassification(trainDim, seed, codeModule, codePresentation, mode):
    X_train, X_test, FS_TMA1_train, FS_TMA1_test, FS_TMA2_train, FS_TMA2_test, FS_TMA3_train, FS_TMA3_test, FS_TMA4_train, FS_TMA4_test, FS_TMA5_train, FS_TMA5_test, FS_TMAF_train, FS_TMAF_test = splitData(trainDim, seed, codeModule, codePresentation, mode)
    
    tree_TMA1 = dtClassifier("DT_TMA1", seed, X_train, FS_TMA1_train, X_test, FS_TMA1_test)
    tree_TMA2 = dtClassifier("DT_TMA2", seed, X_train, FS_TMA2_train, X_test, FS_TMA2_test)
    tree_TMA3 = dtClassifier("DT_TMA3", seed, X_train, FS_TMA3_train, X_test, FS_TMA3_test)
    tree_TMA4 = dtClassifier("DT_TMA4", seed, X_train, FS_TMA4_train, X_test, FS_TMA4_test)
    tree_TMA5 = dtClassifier("DT_TMA5", seed, X_train, FS_TMA5_train, X_test, FS_TMA5_test)
    tree_TMAF = dtClassifier("DT_TMAF", seed, X_train, FS_TMAF_train, X_test, FS_TMAF_test)

    return (tree_TMA1, tree_TMA2, tree_TMA3, tree_TMA4, tree_TMA5, tree_TMAF)

def makeStats(treeStruct, mode):
    
    feat_importances = getFeatureImportances(treeStruct.tree.feature_names_in_, treeStruct.tree.feature_importances_)

    feat_importances.plot(title=treeStruct.name, kind='bar', figsize=(9,7))
    plt.savefig("stats and plots/Decision Trees/"+ mode + "/" + mode + "_" + treeStruct.name +'_feat_importances.svg', format='svg')
    tree.plot_tree(treeStruct.tree, feature_names=treeStruct.tree.feature_names_in_, class_names=['fail', 'pass','dist'])
    plt.savefig("stats and plots/Decision Trees/"+ mode + "/" + mode + "_" + treeStruct.name +'_TreeGraph.svg', format='svg')

    with open("stats and plots/Decision Trees/"+ mode + "/" + mode + "_" + treeStruct.name + "_textfile.txt","w+") as f:
        f.write(np.array2string(feat_importances.index.values, separator=', ', max_line_width=150))
        f.write("\n")
        f.write(np.array2string(feat_importances.values*100, separator=', ', max_line_width=150))
        f.write("\n")
        f.write(np.array2string(confusion_matrix(treeStruct.test, treeStruct.prediction), separator=', ', max_line_width=150))
        f.write("\n")
        f.write(classification_report(treeStruct.test, treeStruct.prediction, zero_division=0))
        f.close()
        
def makeBench(treeStruct, X_bench, y_bench):
    
    
    X_pred = treeStruct.tree.predict(X_bench)
    print(classification_report(X_pred, y_bench, zero_division=0))  
