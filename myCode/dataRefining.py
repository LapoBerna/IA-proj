__package__ = "dataRefining"

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
#courses = pd.read_csv('sklearn-env/myCode/anonymisedData/courses.csv')
#studentRegistration = pd.read_csv('sklearn-env/myCode/anonymisedData/studentRegistration.csv')
#studentVle = pd.read_csv('sklearn-env/myCode/anonymisedData/studentVle.csv')
#vle = pd.read_csv('sklearn-env/myCode/anonymisedData/vle.csv')


def getRefinedStudents(codeModule, codePresentation, mode):

#csv load
    assessments = pd.read_csv('myCode/anonymisedData/assessments.csv')
    studentAssessment = pd.read_csv('myCode/anonymisedData/studentAssessment.csv')
    studentInfo = pd.read_csv('myCode/anonymisedData/studentInfo.csv')

#students refine
    moduleAssessments = assessments[(assessments["assessment_type"] == "TMA") & (assessments["code_module"] == codeModule) & (assessments["code_presentation"] == codePresentation)]#compiti somministrati dal tutor del corso interessato
    moduleStudentAssessment = pd.merge(moduleAssessments, studentAssessment, on='id_assessment', how='left')# compiti consegnati dagli studenti del corso interessato
    examCountByStud = moduleStudentAssessment.groupby(['id_student']).agg(exams_count = ('id_assessment', 'count'))# conteggio consegne per studente
    finalStudents = pd.merge(examCountByStud[(examCountByStud["exams_count"] == 5)], studentInfo[(studentInfo["final_result"]!= "Withdrawn") & (studentInfo["code_module"] == codeModule) & (studentInfo["code_presentation"] == codePresentation)], on="id_student", how="inner")#studenti mai ritirati e che hanno consegnato ogni compito assegnato del corso interessato
    finalStudents = finalStudents.drop(columns=['exams_count', 'code_module','code_presentation','num_of_prev_attempts','studied_credits'])
   
#converting categorical data to numerical
    if mode == 'categoricalValues':
        #one-hot 
        highest_education =['EDU_No Formal quals', 'EDU_Lower Than A Level', 'EDU_A Level or Equivalent', 'EDU_HE Qualification', 'EDU_Post Graduate Qualification']
        imd_band = ["IMD_0-10%","IMD_10-20", "IMD_20-30%", "IMD_30-40%", "IMD_40-50%", "IMD_60-70%","IMD_70-80%", "IMD_80-90%", "IMD_90-100%"]
        age_band = ["AGE_0-35", "AGE_35-55", "AGE_55<="]
        region = ["REGION_East Anglian Region", "REGION_East Midlands Region", "REGION_Ireland", "REGION_London Region", "REGION_North Region", "REGION_North Western Region", "REGION_Scotland", "REGION_South East Region", "REGION_South Region", "REGION_South West Region", "REGION_Wales", "REGION_West Midlands Region", "REGION_Yorkshire Region"]
        encoded_he = pd.get_dummies(finalStudents["highest_education"], prefix='EDU', prefix_sep='_')
        encoded_he = encoded_he.T.reindex(highest_education, fill_value=0).T.astype(int)
        encoded_ib = pd.get_dummies(finalStudents["imd_band"], prefix='IMD', prefix_sep='_')
        encoded_ib = encoded_ib.T.reindex(imd_band, fill_value=0).T.astype(int)
        encoded_ab = pd.get_dummies(finalStudents["age_band"], prefix='AGE', prefix_sep='_')
        encoded_ab = encoded_ab.T.reindex(age_band, fill_value=0).T.astype(int)
        encoded_r = pd.get_dummies(finalStudents["region"], prefix='REGION', prefix_sep='_')
        encoded_r = encoded_r.T.reindex(region, fill_value=0).T.astype(int)
        finalStudents = pd.concat([finalStudents, encoded_he, encoded_ib, encoded_ab, encoded_r], axis=1)
        #already boolean
        finalStudents['gender'] = finalStudents['gender'].astype('category')
        finalStudents['gender_Codes'] = finalStudents['gender'].cat.codes
        finalStudents['disability'] = finalStudents['disability'].astype('category')
        finalStudents['disability_Codes'] = finalStudents['disability'].cat.codes
        finalStudents['final_result'] = finalStudents['final_result'].astype('category').cat.reorder_categories(['Fail', 'Pass', 'Distinction'], ordered=True)
        finalStudents['final_result_Codes'] = finalStudents['final_result'].cat.codes
        finalStudents = finalStudents.drop(columns=['gender', 'disability', 'highest_education', 'imd_band', 'age_band', 'region', 'final_result'])
    elif mode == 'hybridValues':
        
        region = ["REGION_East Anglian Region", "REGION_East Midlands Region", "REGION_Ireland", "REGION_London Region", "REGION_North Region", "REGION_North Western Region", "REGION_Scotland", "REGION_South East Region", "REGION_South Region", "REGION_South West Region", "REGION_Wales", "REGION_West Midlands Region", "REGION_Yorkshire Region"]
        encoded_r = pd.get_dummies(finalStudents["region"], prefix='REGION', prefix_sep='_')
        encoded_r = encoded_r.T.reindex(region, fill_value=0).T.astype(int)
        finalStudents = pd.concat([finalStudents, encoded_r], axis=1)

        finalStudents['highest_education'] = finalStudents['highest_education'].astype('category').cat.set_categories(['No Formal quals', 'Lower Than A Level', 'A Level or Equivalent', 'HE Qualification', 'Post Graduate Qualification'], ordered=True)
        finalStudents['highest_education_Codes'] = finalStudents['highest_education'].cat.codes
        finalStudents['imd_band'] = finalStudents['imd_band'].astype('category')
        finalStudents['imd_band_Codes'] = finalStudents['imd_band'].cat.codes
        finalStudents['age_band'] = finalStudents['age_band'].astype('category')
        finalStudents['age_band_Codes'] = finalStudents['age_band'].cat.codes
        finalStudents['gender'] = finalStudents['gender'].astype('category')
        finalStudents['gender_Codes'] = finalStudents['gender'].cat.codes
        finalStudents['disability'] = finalStudents['disability'].astype('category')
        finalStudents['disability_Codes'] = finalStudents['disability'].cat.codes
        finalStudents['final_result'] = finalStudents['final_result'].astype('category').cat.reorder_categories(['Fail', 'Pass', 'Distinction'], ordered=True)
        finalStudents['final_result_Codes'] = finalStudents['final_result'].cat.codes
        finalStudents = finalStudents.drop(columns=['gender', 'disability', 'highest_education', 'imd_band', 'age_band', 'region', 'final_result']) 
    else:
        finalStudents['gender'] = finalStudents['gender'].astype('category')
        finalStudents['gender_Codes'] = finalStudents['gender'].cat.codes
        finalStudents['disability'] = finalStudents['disability'].astype('category')
        finalStudents['disability_Codes'] = finalStudents['disability'].cat.codes
        finalStudents['highest_education'] = finalStudents['highest_education'].astype('category').cat.set_categories(['No Formal quals', 'Lower Than A Level', 'A Level or Equivalent', 'HE Qualification', 'Post Graduate Qualification'], ordered=True)
        finalStudents['highest_education_Codes'] = finalStudents['highest_education'].cat.codes
        finalStudents['imd_band'] = finalStudents['imd_band'].astype('category')
        finalStudents['imd_band_Codes'] = finalStudents['imd_band'].cat.codes
        finalStudents['age_band'] = finalStudents['age_band'].astype('category')
        finalStudents['age_band_Codes'] = finalStudents['age_band'].cat.codes
        finalStudents['region'] = finalStudents['region'].astype('category')
        finalStudents['region_Codes'] = finalStudents['region'].cat.codes
        finalStudents['final_result'] = finalStudents['final_result'].astype('category').cat.reorder_categories(['Fail', 'Pass', 'Distinction'], ordered=True)
        finalStudents['final_result_Codes'] = finalStudents['final_result'].cat.codes
        finalStudents = finalStudents.drop(columns=['gender', 'disability', 'highest_education', 'imd_band', 'age_band', 'region', 'final_result'])

#getting tests results
    assessmentsByType = pd.merge(moduleAssessments,studentAssessment, on='id_assessment', how='left')
    testResults = pd.merge(finalStudents, assessmentsByType, on='id_student', how='left')
    testResults = testResults.pivot(index='id_student', columns=['assessment_type', 'id_assessment'], values='score' )
    testResults.columns=['TMA1', 'TMA2', 'TMA3','TMA4', 'TMA5']
    finalStudents = pd.merge(finalStudents,testResults, on='id_student', how='left')

#converting scores data to numerical categories
    finalStudents['TMA1'] = pd.cut(finalStudents["TMA1"], bins=[1, 54, 84, 100], labels=['0', '1', '2'])
    finalStudents['TMA2'] = pd.cut(finalStudents["TMA2"], bins=[1, 54, 84, 100], labels=['0', '1', '2'])
    finalStudents['TMA3'] = pd.cut(finalStudents["TMA3"], bins=[1, 54, 84, 100], labels=['0', '1', '2'])
    finalStudents['TMA4'] = pd.cut(finalStudents["TMA4"], bins=[1, 54, 84, 100], labels=['0', '1', '2'])
    finalStudents['TMA5'] = pd.cut(finalStudents["TMA5"], bins=[1, 54, 84, 100], labels=['0', '1', '2'])

#dividing input and outputs
    if mode == 'categoricalValues':
        FSData = finalStudents.iloc[:,1:33]
        FS_TMAF = finalStudents.iloc[:,33]
        FS_TMA1 = finalStudents.iloc[:,34]
        FS_TMA2 = finalStudents.iloc[:,35]
        FS_TMA3 = finalStudents.iloc[:,36]
        FS_TMA4 = finalStudents.iloc[:,37]
        FS_TMA5 = finalStudents.iloc[:,38]
    elif mode == 'hybridValues':
        FSData = finalStudents.iloc[:,1:19]
        FS_TMAF = finalStudents.iloc[:,19]
        FS_TMA1 = finalStudents.iloc[:,20]
        FS_TMA2 = finalStudents.iloc[:,21]
        FS_TMA3 = finalStudents.iloc[:,22]
        FS_TMA4 = finalStudents.iloc[:,23]
        FS_TMA5 = finalStudents.iloc[:,24]
    else:
        FSData = finalStudents.iloc[:,1:7]
        FS_TMAF = finalStudents.iloc[:,7]
        FS_TMA1 = finalStudents.iloc[:,8]
        FS_TMA2 = finalStudents.iloc[:,9]
        FS_TMA3 = finalStudents.iloc[:,10]
        FS_TMA4 = finalStudents.iloc[:,11]
        FS_TMA5 = finalStudents.iloc[:,12]
    
    return (FSData, FS_TMA1, FS_TMA2, FS_TMA3, FS_TMA4, FS_TMA5, FS_TMAF) 

def splitData(trainSize, key, codeModule, codePresentation, mode):
 
    X, FS_TMA1, FS_TMA2, FS_TMA3, FS_TMA4, FS_TMA5, FS_TMAF = getRefinedStudents(codeModule, codePresentation, mode)
    X_train, X_test, FS_TMA1_train, FS_TMA1_test, FS_TMA2_train, FS_TMA2_test, FS_TMA3_train, FS_TMA3_test, FS_TMA4_train, FS_TMA4_test, FS_TMA5_train, FS_TMA5_test, FS_TMAF_train, FS_TMAF_test = train_test_split(X, FS_TMA1, FS_TMA2, FS_TMA3, FS_TMA4, FS_TMA5, FS_TMAF, random_state=key, train_size=trainSize)
    
    return (X_train, X_test, FS_TMA1_train, FS_TMA1_test, FS_TMA2_train, FS_TMA2_test, FS_TMA3_train, FS_TMA3_test, FS_TMA4_train, FS_TMA4_test, FS_TMA5_train, FS_TMA5_test, FS_TMAF_train, FS_TMAF_test)


