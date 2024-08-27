import pandas as pd

def getFeatureImportances(names, values):

    feat_importances = pd.DataFrame(values, index=names, columns=["Importance"])
    rSum, aSum, iSum, eSum = 0, 0 ,0 ,0

    for i in range(len(names)):

        if "REGION" in names[i]:
            rSum += values[i]
            feat_importances = feat_importances.drop(index=names[i])
        elif "AGE" in names[i]:
            aSum += values[i]
            feat_importances = feat_importances.drop(index=names[i])
        elif "IMD" in names[i]:
            iSum += values[i]
            feat_importances = feat_importances.drop(index=names[i])
        elif "EDU" in names[i]:
            eSum += values[i]
            feat_importances = feat_importances.drop(index=names[i])
    if rSum:
        feat_importances.loc['regions'] = [rSum]
    if aSum:
        feat_importances.loc['age_band'] = [aSum]
    if iSum:
        feat_importances.loc['imd_band'] = [iSum]
    if eSum:
        feat_importances.loc['highest_education'] = [eSum]
    
    feat_importances = feat_importances.sort_index()

    
    return feat_importances