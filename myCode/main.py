
import decisionTree as dt
import randomForest as rf
from dataRefining import getRefinedStudents 
import os

def checkDirs():
    
    script_dir = os.path.dirname(os.path.dirname(__file__))
    plots_dir = os.path.join(script_dir, 'stats and plots/')
    dt_dir = os.path.join(plots_dir, 'Decision Trees/')
    rf_dir = os.path.join(plots_dir, 'Random Forests/')
    hdt_dir = os.path.join(dt_dir, 'hybridValues/')
    cdt_dir = os.path.join(dt_dir, 'categoricalValues/')
    ndt_dir = os.path.join(dt_dir, 'numericalValues/')
    hrf_dir = os.path.join(rf_dir, 'hybridValues/')
    crf_dir = os.path.join(rf_dir, 'categoricalValues/')
    nrf_dir = os.path.join(rf_dir, 'numericalValues/')
    if not os.path.isdir(plots_dir):
        os.makedirs(plots_dir)
        os.makedirs(dt_dir)
        os.makedirs(rf_dir)
        os.makedirs(hdt_dir)
        os.makedirs(cdt_dir)
        os.makedirs(ndt_dir)
        os.makedirs(hrf_dir)
        os.makedirs(crf_dir)
        os.makedirs(nrf_dir)
    if not os.path.isdir(dt_dir) :
        os.makedirs(dt_dir)
        os.makedirs(hdt_dir)
        os.makedirs(cdt_dir)
        os.makedirs(ndt_dir)
    if not os.path.isdir(rf_dir) :
        os.makedirs(rf_dir)
        os.makedirs(hrf_dir)
        os.makedirs(crf_dir)
        os.makedirs(nrf_dir)
    if not os.path.isdir(hdt_dir) :
        os.makedirs(hdt_dir)
    if not os.path.isdir(cdt_dir) :
        os.makedirs(cdt_dir)
    if not os.path.isdir(ndt_dir) :
        os.makedirs(ndt_dir)
    if not os.path.isdir(hrf_dir) :
        os.makedirs(hrf_dir)
    if not os.path.isdir(crf_dir) :
        os.makedirs(crf_dir)
    if not os.path.isdir(nrf_dir) :
        os.makedirs(nrf_dir)


checkDirs()

mode = 'numericalValues'   #   'categoricalValues'   'numericalValues'     'hybridValues'
seed = 1

tree_TMA1, tree_TMA2, tree_TMA3, tree_TMA4, tree_TMA5, tree_TMAF = dt.decisionTreeClassification(200, seed, "AAA", "2013J", mode)
forest_TMA1, forest_TMA2, forest_TMA3, forest_TMA4, forest_TMA5, forest_TMAF = rf.randomForestClassification(200, seed, "AAA", "2013J", mode)


X_bench, FS_TMA1, FS_TMA2, FS_TMA3, FS_TMA4, FS_TMA5, FS_TMAF = getRefinedStudents("AAA", "2014J", mode)




dt.makeStats(tree_TMA1, mode)
dt.makeStats(tree_TMA2, mode)
dt.makeStats(tree_TMA3, mode)
dt.makeStats(tree_TMA4, mode)
dt.makeStats(tree_TMA5, mode)
dt.makeStats(tree_TMAF, mode)

rf.makeStats(forest_TMA1, seed, mode)
rf.makeStats(forest_TMA2, seed, mode)
rf.makeStats(forest_TMA3, seed, mode)
rf.makeStats(forest_TMA4, seed, mode)
rf.makeStats(forest_TMA5, seed, mode)
rf.makeStats(forest_TMAF, seed, mode)