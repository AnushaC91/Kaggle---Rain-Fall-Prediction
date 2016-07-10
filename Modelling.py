import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
from sklearn.metrics import mean_absolute_error
from sklearn import ensemble
import numpy as np
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from xgboost import XGBRegressor;
from sklearn.preprocessing import Imputer
from scipy.optimize import minimize
from sklearn.svm import SVR

TRAINING_INPUT = "/tmp/sampleTrainData.csv";
TEST_INPUT = "/tmp/sampleTestData.csv";

#TRAINING_INPUT = "cleanTD1.csv";
#TEST_INPUT = "cleanTestData1.csv";

def ImputeAndGetFinalTrainTestData(train,test):
    X_train = train[:,:-1];
    y_train = train[:,-1];
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(X_train);
    X_train = imp.transform(X_train);
    X_test = imp.transform(test.as_matrix());
    return (X_train,y_train,X_test)

def ValidateTrainTestErrorsWithDifferentModels(cvX_train, cvX_test, cvy_train, cvy_test,X_train,y_train,X_test):
    clfs = list()
    cvClfs = list()

    print "Building RF1"
    rfShortCV = ensemble.RandomForestRegressor(min_samples_split=50,n_estimators=1000, max_depth=None, min_samples_leaf=50, max_features="auto", n_jobs=-1, random_state=0)
    rfShort = ensemble.RandomForestRegressor(min_samples_split=50,n_estimators=1000, max_depth=None, min_samples_leaf=50, max_features="auto", n_jobs=-1, random_state=0)
    rfShortCV.fit(cvX_train, cvy_train);
    print 'RF1 CV Results :',mean_absolute_error(cvy_test,rfShortCV.predict(cvX_test))
    pd.DataFrame({"Actual":cvy_test, "Predicted":rfShortCV.predict(cvX_test)}).to_csv("snehaRF.csv", index=False,header=True);
    rfShort.fit(X_train,y_train)
    cvClfs.append(rfShortCV)
    clfs.append(rfShort)
    pd.DataFrame({"ID":out_id, "Expected":rfShort.predict(X_test)}).to_csv("subRF1.csv", index=False,header=True);

    print "Building SVM"
    clfSVRCV = SVR(C=10.0)
    clfSVR = SVR(C=10.0)
    clfSVRCV.fit(cvX_train, cvy_train);
    print 'SVM CV Results :',mean_absolute_error(cvy_test,clfSVRCV.predict(cvX_test))
    pd.DataFrame({"Actual":cvy_test, "Predicted":clfSVRCV.predict(cvX_test)}).to_csv("snehaSVR.csv", index=False,header=True);

    print "Building RF2"
    rfLongCV = ensemble.RandomForestRegressor(min_samples_split=200,n_estimators=1000, max_depth=7, min_samples_leaf=200, max_features="auto", n_jobs=4, random_state=0)
    rfLong = ensemble.RandomForestRegressor(min_samples_split=200,n_estimators=1000, max_depth=7, min_samples_leaf=200, max_features="auto", n_jobs=4, random_state=0)
    rfLongCV.fit(cvX_train, cvy_train);
    print 'RF2 CV Results :',mean_absolute_error(cvy_test,rfLongCV.predict(cvX_test))
    rfLong.fit(X_train,y_train)
    cvClfs.append(rfLongCV)
    clfs.append(rfLong)
    pd.DataFrame({"ID":out_id, "Expected":rfLong.predict(X_test)}).to_csv("subRF2.csv", index=False,header=True);


    print "Building GB1"
    regGBCV1 = ensemble.GradientBoostingRegressor(min_samples_split=50,n_estimators=1000, max_depth=None, min_samples_leaf=50, max_features="auto", subsample=0.6, learning_rate=0.01, random_state=0,loss='lad')
    regGBCV1.fit(cvX_train, cvy_train);
    print 'GB1 CV Results :',mean_absolute_error(cvy_test,regGBCV1.predict(cvX_test))
    regGB1 = ensemble.GradientBoostingRegressor(min_samples_split=50,n_estimators=1000, max_depth=None, min_samples_leaf=50, max_features="auto", subsample=0.6, learning_rate=0.01, random_state=0,loss='lad')
    regGB1.fit(X_train,y_train)
    cvClfs.append(regGBCV1)
    clfs.append(regGB1)
    pd.DataFrame({"ID":out_id, "Expected":regGB1.predict(X_test)}).to_csv("subGB1.csv", index=False,header=True);


    print 'Building GB2'
    regGBCV2 = ensemble.GradientBoostingRegressor(min_samples_split=50,n_estimators=1000, max_depth=7, min_samples_leaf=200, max_features="auto", subsample=0.6, learning_rate=0.01, random_state=0,loss='lad')
    regGBCV2.fit(cvX_train, cvy_train);
    print 'GB2 CV Results :',mean_absolute_error(cvy_test,regGBCV2.predict(cvX_test))
    regGB2 = ensemble.GradientBoostingRegressor(min_samples_split=50,n_estimators=1000, max_depth=7, min_samples_leaf=200, max_features="auto", subsample=0.6, learning_rate=0.01, random_state=0,loss='lad')
    regGB2.fit(X_train,y_train)
    cvClfs.append(regGBCV2)
    clfs.append(regGB2)
    pd.DataFrame({"ID":out_id, "Expected":regGB2.predict(X_test)}).to_csv("subGB2.csv", index=False,header=True);


    print 'Feature Importances RF1:',sorted(zip(map(lambda x: round(x, 4), rfShort.feature_importances_), df_final.columns),reverse=True);
    print 'Feature Importances GB1:',sorted(zip(map(lambda x: round(x, 4), regGB1.feature_importances_), df_final.columns),reverse=True);
    print 'Feature Importances RF2:',sorted(zip(map(lambda x: round(x, 4), rfLong.feature_importances_), df_final.columns),reverse=True);
    print 'Feature Importances GB2:',sorted(zip(map(lambda x: round(x, 4), regGB2.feature_importances_), df_final.columns),reverse=True);

    print "Building XGB1"
    xgbCV1 = xgb.XGBRegressor(n_estimators=3000, nthread=-1, max_depth=None,
                        learning_rate=0.01, silent=True, subsample=0.8, colsample_bytree=0.7)
    xgbCV1.fit(cvX_train, cvy_train);
    xgb1 = xgb.XGBRegressor(n_estimators=3000, nthread=-1, max_depth=None,
                        learning_rate=0.01, silent=True, subsample=0.8, colsample_bytree=0.7)
    xgb1.fit(X_train,y_train);
    print 'XGB1 Model CV :',mean_absolute_error(cvy_test,xgbCV1.predict(cvX_test));
    cvClfs.append(xgbCV1)
    clfs.append(xgb1)
    pd.DataFrame({"ID":out_id, "Expected":xgb1.predict(X_test)}).to_csv("subXGB1.csv", index=False,header=True);



    print "Building XGB2"
    params = {}
    params["objective"] = "reg:linear"
    params["learning_rate"] = 0.005
    params["min_child_weight"] = 6
    params["subsample"] = 0.7
    params["colsample_bytree"] = 0.75
    params["silent"] = 1
    params["max_depth"] = 7
    params["n_estimators"] = 3000
    params['gamma'] = 1.25
    params['nthread'] = -1
    print 'XGBoost Training Process Started'
    xgbCV2 = XGBRegressor(**params);
    xgbCV2.fit(cvX_train, cvy_train);
    print 'XGB Model CV :',mean_absolute_error(cvy_test,xgbCV2.predict(cvX_test));
    xgb2 = XGBRegressor(**params);
    xgb2.fit(X_train,y_train);
    cvClfs.append(xgbCV2)
    clfs.append(xgb2)
    pd.DataFrame({"ID":out_id, "Expected":xgb2.predict(X_test)}).to_csv("subXGB2.csv", index=False,header=True);


    # Return the cross validated models and the actual fitted models separately.
    return [clfs,cvClfs];


def mae_loss_function(weights,predictions,test_y):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = 0
    for weight, prediction in zip(weights, predictions):
            final_prediction += weight*prediction

    return mean_absolute_error(test_y, final_prediction)

if __name__=='__main__':
    train = pd.read_csv(TRAINING_INPUT);
    test =  pd.read_csv(TEST_INPUT);
    train_final = train.drop("Id",axis=1,inplace=False)
    df_final = train_final;
    global out_id;
    out_id = test["Id"].as_matrix() + 1;
    input_test_data = test.drop("Id",axis=1,inplace=False);
    input_train_data = train_final.as_matrix();

    (X_train,y_train,X_test) = ImputeAndGetFinalTrainTestData(input_train_data,input_test_data)
    cvX_train, cvX_test, cvy_train, cvy_test = cross_validation.train_test_split(X_train,y_train, test_size=0.25, random_state=0)
    # This function builds models and hence doesnt need any test data
    [clfs,cvClfs] = ValidateTrainTestErrorsWithDifferentModels(cvX_train, cvX_test, cvy_train, cvy_test,X_train,y_train,X_test);

    print 'Clfs are :',clfs
    # Optimizing the Weights of the predictions
    print ' Finding the Optimal Weights of ensemble.'
    predictions = []
    for clf in cvClfs:
        predictions.append(clf.predict(cvX_test))

    clf_count = len(cvClfs)
    starting_values = [1.0/clf_count]*clf_count
    cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
    #our weights are bound between 0 and 1
    bounds = [(0,1)]*len(clfs)
    ensemble_weights = minimize(mae_loss_function, starting_values,(predictions,cvy_test),method="SLSQP", bounds=tuple(bounds), constraints=cons,
                               options = {'disp':True,'eps' : 0.05})
    res = ensemble_weights;

    print('Ensamble Score: {best_score}'.format(best_score=res['fun']))
    print('Best Weights: {weights}'.format(weights=res['x']))

    final_weights = res['x'];
    #Calculating final MAE
    final_prediction = 0
    for weight,prediction in zip(final_weights,predictions):
            final_prediction += weight*prediction;
    print 'Final MAE after ensemble aggregation :',mean_absolute_error(cvy_test,final_prediction)


    #Calculating the actual Results/predictions.
    test_data_final_prediction = 0
    for weight,clf in zip(final_weights,clfs):
            test_data_final_prediction += clf.predict(X_test)

    out_df = pd.DataFrame({"ID":out_id, "Expected":test_data_final_prediction})
    out_df.to_csv("submission.csv", index=False,header=True);