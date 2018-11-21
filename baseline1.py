# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 21:12:04 2018

@author: Administrator
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
def tpr_weight_funtion(y_true,y_predict):
    d = pd.DataFrame()
    d['prob'] = list(y_predict)
    d['y'] = list(y_true)
    d = d.sort_values(['prob'], ascending=[0])
    y = d.y
    PosAll = pd.Series(y).value_counts()[1]
    NegAll = pd.Series(y).value_counts()[0]
    pCumsum = d['y'].cumsum()
    nCumsum = np.arange(len(y)) - pCumsum + 1
    pCumsumPer = pCumsum / PosAll
    nCumsumPer = nCumsum / NegAll
    TR1 = pCumsumPer[abs(nCumsumPer-0.001).idxmin()]
    TR2 = pCumsumPer[abs(nCumsumPer-0.005).idxmin()]
    TR3 = pCumsumPer[abs(nCumsumPer-0.01).idxmin()]
    return 'TC_AUC',0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3,True

op_train = pd.read_csv('operation_train_new.csv')
trans_train = pd.read_csv('transaction_train_new.csv')

op_test = pd.read_csv('operation_round1_new.csv')
trans_test = pd.read_csv('transaction_round1_new.csv')
y = pd.read_csv('tag_train_new.csv')
sub = pd.read_csv('sub.csv')

op_train.columns
y['Tag'].value_counts()
op_columns = op_train.columns.tolist()
trans_columns = trans_train.columns.tolist()
same_columns = list(set(op_columns).intersection(set(trans_columns)))

#########查看每个特征的缺失个数
null_column=pd.DataFrame()
null_column['trans_train']=trans_train.isnull().sum()/len(trans_train['UID'])
null_column['trans_test']=trans_test.isnull().sum()/len(trans_test['UID'])
null_column

op_null_column = pd.DataFrame()
op_null_column ['op_train']=op_train.isnull().sum()/len(op_train['UID'])
op_null_column ['op_test']=op_test.isnull().sum()/len(op_test['UID'])
op_null_column

#############合并训练集和测试集
op_train['month'] = 7
op_test['month'] = 8
trans_train['month'] = 7
trans_test['month'] = 8
          
import time
import datetime

def get_time_unix(data):
    data['month'] = data.month.apply(lambda x:str(x))
    data['day'] = data.day.apply(lambda x:str(x))
    data['time_stamp'] = '2018-'+data['month']+'-'+data['day']+' '+data['time']
    data['time_stamp'] = data.time_stamp.apply(lambda x:datetime.datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))
    data['time_stamp'] = data.time_stamp.apply(lambda x:int(time.mktime(x.timetuple())))
    return data

op_train = get_time_unix(op_train)
op_test = get_time_unix(op_test)
trans_train = get_time_unix(trans_train)
trans_test = get_time_unix(trans_test)

#op_test['day'] = op_test['day']+32
#trans_test['day'] = trans_test['day']+32

op_train =pd.concat([op_train,op_test],axis=0,ignore_index=True)
trans_train = pd.concat([trans_train,trans_test],axis=0,ignore_index=True)


##########查看每一行的缺失个数作为单独特征
u = op_train.isnull().sum(axis=1)
def get_row_null(data,vname):
    data[vname+'lv_row_null']=data.isnull().sum(axis=1)/len(data.columns.tolist())
    data[vname+'row_null']=data.isnull().sum(axis=1)
    return data

op_train = get_row_null(op_train,'op_train')
trans_train = get_row_null(trans_train,'trans_train')

##################删除缺失值太多的特征
for feature in op_columns:
    if(op_null_column.ix[feature,'op_train']>=0.5):
        op_train.drop(feature,axis=1,inplace=True)
    
for feature in trans_columns:
    if(null_column.ix[feature,'trans_train']>=0.5):
        trans_train.drop(feature,axis=1,inplace=True)

##################查看特征的类型
print("oP_train_feature")
for feature in op_train.columns.tolist():
    print(feature,' ',op_train[feature].dtype)

print("trnas_train_feature"+"\n")
for feature in trans_train.columns.tolist():
    print(feature,' ',trans_train[feature].dtype)

###################缺失值填充为-1
op_train = op_train.fillna(-1)
trans_train = trans_train.fillna(-1)

####################对特征进行labelencoder

###############
new_op_column = op_train.columns.tolist()
new_trans_column = trans_train.columns.tolist()

op_drop =['UID','day','success','time','time_stamp','month','op_trainlv_row_null','op_trainrow_null']
op_fea = []
for feature in new_op_column:
    if feature not in op_drop:
        op_fea.append(feature)

trans_drop=['UID','day','time','trans_amt','bal','month','time_stamp','trans_trainlv_row_null','trans_trainrow_null']
trans_fea = []
for feature in new_trans_column:
    if feature not in trans_drop:
        trans_fea.append(feature)

u = trans_train[['UID']]
u['UID_'+'count'] = 1
u = u.groupby(['UID']).agg('sum').reset_index()

u = op_train.groupby(['UID','mode'])['version'].count().reset_index().rename(columns={'version':'version_co'})
u1 = op_train.groupby(['UID'])['mode'].count().reset_index().rename(columns={'mode':'mode_co'})


def get_low_fea(op,op_fea,trans,trans_fea,data):
    #########单一uid_count特征
    u = op[['UID']]
    u['op_UID_'+'count'] = 1
    u = u.groupby(['UID']).agg('sum').reset_index()
    data = pd.merge(data,u,on='UID',how='left')
    
    
    u = trans[['UID']]
    u['trans_UID_'+'count'] = 1
    u  = u.groupby(['UID']).agg('sum').reset_index()
    data = pd.merge(data,u,on='UID',how='left')
    
    ###########uid关联特征count
    for feature in op_fea:
        u = op.groupby(['UID'])[feature].count().reset_index().rename(columns={feature:'op_'+feature+'_count'})
        data = pd.merge(data,u,on='UID',how='left')
        u = op.groupby(['UID'])[feature].nunique().reset_index().rename(columns={feature:'op_'+feature+'_nun'})
        data = pd.merge(data,u,on='UID',how='left')
    
    for feature in trans_fea:
         u = trans.groupby(['UID'])[feature].count().reset_index().rename(columns={feature:'trans_'+feature+'_count'})
         data = pd.merge(data,u,on='UID',how='left')
         u = trans.groupby(['UID'])[feature].nunique().reset_index().rename(columns={feature:'trans_'+feature+'_nun'})
         data = pd.merge(data,u,on='UID',how='left')
    
    return data

def get_sta_fea(trans,trans_fea,data):
    
    ###############数值特征的统计特征
    
    for feature in trans_fea:
        u = trans.groupby(['UID'])[feature].max().reset_index().rename(columns={feature:'trans_'+feature+'_max'})
        data = pd.merge(data,u,on='UID',how='left')
        u = trans.groupby(['UID'])[feature].min().reset_index().rename(columns={feature:'trans_'+feature+'_min'})
        data = pd.merge(data,u,on='UID',how='left')
        u = trans.groupby(['UID'])[feature].mean().reset_index().rename(columns={feature:'trans_'+feature+'_mean'})
        data = pd.merge(data,u,on='UID',how='left')
        u = trans.groupby(['UID'])[feature].sum().reset_index().rename(columns={feature:'trans_'+feature+'_sum'})
        data = pd.merge(data,u,on='UID',how='left')
        u = trans.groupby(['UID'])[feature].std().reset_index().rename(columns={feature:'trans_'+feature+'_std'})
        data = pd.merge(data,u,on='UID',how='left')
    
    return data

data = y.copy()
op_sta_fea=['op_trainlv_row_null','op_trainrow_null']
trans_sta_fea=['trans_amt','bal','trans_trainlv_row_null','trans_trainrow_null']
data = get_low_fea(op_train,op_fea,trans_train,trans_fea,data)
data = get_sta_fea(trans_train,trans_sta_fea,data)
data = get_sta_fea(op_train,op_sta_fea,data)

data_test = sub.copy()
data_test = get_low_fea(op_train,op_fea,trans_train,trans_fea,data_test)
data_test = get_sta_fea(trans_train,trans_sta_fea,data_test)
data_test = get_sta_fea(op_train,op_sta_fea,data_test)
    



skf = StratifiedKFold(n_splits=5, random_state=2018, shuffle=True)
best_score = []

label = data['Tag']
train = data.drop(['UID','Tag'],axis=1)

test = data_test.drop(['UID','Tag'],axis=1)

oof_preds = np.zeros(train.shape[0])
sub_preds = np.zeros(test.shape[0])


from sklearn import model_selection
X_train,X_test,y_train,y_test = model_selection.train_test_split(train,label,test_size=0.1,random_state=2018)



lgb0 = lgb.LGBMClassifier(boosting_type='gbdt', 
                               num_leaves=7,
#                               reg_alpha=3, 
#                               reg_lambda=5, 
                               max_depth=-1,
                               n_estimators=10000, 
                               objective='binary', 
                               subsample=0.9, 
                               colsample_bytree=0.8, 
#                               subsample_freq=1, 
                               learning_rate=0.05,
                               seed=2018, 
#                               n_jobs=16, 
#                               min_child_weight=4,
#                               min_child_samples=5, 
#                               min_split_gain=0
                               )

############寻找超参数
#from sklearn.grid_search import GridSearchCV
#params = {
#        'num_leaves':[15,31,63]
#        }
#
#gsearch = GridSearchCV(estimator=lgb0,param_grid=params,scoring='roc_auc',iid=False,cv=5,verbose=1)
#gsearch.fit(train,label)
#gsearch.grid_scores_,gsearch.best_params_,gsearch.best_score_


lgb_model = lgb0.fit(X_train,y_train,eval_set=[(X_test,y_test)],eval_metric='auc',early_stopping_rounds=50)

best_iter = lgb_model.best_iteration

predictors = [i for i in X_train.columns]
feat_imp = pd.Series(lgb_model.feature_importance(), predictors).sort_values(ascending=False)
print(feat_imp)
print(feat_imp.shape)

############查看test的评估指标值
pred1 = lgb_model.predict_proba(X_test)[:,1]
m = tpr_weight_funtion(y_predict=pred1,y_true=y_test)

print("评估指标为： ",m)

#################对提交的sub进行预测
lgb1 = lgb.LGBMClassifier(boosting_type='gbdt', 
                               num_leaves=7,
#                               reg_alpha=3, 
#                               reg_lambda=5, 
                               max_depth=-1,
                               n_estimators=best_iter, 
                               objective='binary', 
                               subsample=0.9, 
                               colsample_bytree=0.8, 
#                               subsample_freq=1, 
                               learning_rate=0.05,
                               seed=2018, 
#                               n_jobs=16, 
#                               min_child_weight=4,
#                               min_child_samples=5, 
                                 )



lgb_model = lgb1.fit(train,label)

pred2 = lgb_model.predict_proba(test)[:,1]

sub['Tag'] = pred2
sub.to_csv('sub_baseline_%s.csv'%str(m),index=False)    





#sub.to_csv('sub/baseline_%s.csv'%str(m),index=False)    ]
#lgb_train = lgb.Dataset(X_train,y_train)
#lgb_eval = lgb.Dataset(X_test,y_test)
#
#
#
#
#
#
###############交叉验证
#for index,(train_index,test_index) in enumerate(skf.split(train,y_train)):
#    lgb_model.fit(train.iloc[train_index],y_train.iloc[train_index],verbose=1,
#                  eval_set=[(train.iloc[train_index],y_train.iloc[train_index]),(train.iloc[test_index],y_train.iloc[test_index])],early_stopping_rounds=50)
#    
#    best_score.append(lgb_model.best_score['valid_1']['binaray_logloss'])
#    print(best_score)
#    oof_preds[test_index]=lgb_model.predict_proba(train.iloc[test_index],num_iteration=lgb_model.best_iteration)[:,1]
#    
#    test_pred = lgb_model.predict_proba(test,num_iteration=lgb_model.best_iteration)[:,1]
#    sub_preds +=test_pred/5
#
#
#m = tpr_weight_funtion(y_predict=oof_preds,y_true=y_train)
#    
#print(m[1])
#sub = pd.read_csv('input/sub.csv')
#sub['Tag'] = sub_preds
#sub.to_csv('sub/baseline_%s.csv'%str(m),index=False)    
#    
    
    
    
    
    
#    
#    
#    
#    
#
#for index, (train_index, test_index) in enumerate(skf.split(train, label)):
#    lgb_model.fit(train.iloc[train_index], label.iloc[train_index], verbose=50,
#                  eval_set=[(train.iloc[train_index], label.iloc[train_index]),
#                            (train.iloc[test_index], label.iloc[test_index])], early_stopping_rounds=30)
#    best_score.append(lgb_model.best_score_['valid_1']['binary_logloss'])
#    print(best_score)
#    oof_preds[test_index] = lgb_model.predict_proba(train.iloc[test_index], num_iteration=lgb_model.best_iteration_)[:,1]
#
#    test_pred = lgb_model.predict_proba(test, num_iteration=lgb_model.best_iteration_)[:, 1]
#    sub_preds += test_pred / 5
#    #print('test mean:', test_pred.mean())
#    #predict_result['predicted_score'] = predict_result['predicted_score'] + test_pred
#
#m = tpr_weight_funtion(y_predict=oof_preds,y_true=label)
#print(m[1])
#sub = pd.read_csv('input/sub.csv')
#sub['Tag'] = sub_preds
#sub.to_csv('sub/baseline_%s.csv'%str(m),index=False)

    
    
#def get_feature(op,trans,label):
#    for feature in op.columns[2:]:
#        label = label.merge(op.groupby(['UID'])[feature].count().reset_index(),on='UID',how='left')
#        label =label.merge(op.groupby(['UID'])[feature].nunique().reset_index(),on='UID',how='left')
#    
#    for feature in trans.columns[2:]:
#        if trans_train[feature].dtype == 'object':
#            label =label.merge(trans.groupby(['UID'])[feature].count().reset_index(),on='UID',how='left')
#            label =label.merge(trans.groupby(['UID'])[feature].nunique().reset_index(),on='UID',how='left')
#        else:
#            label =label.merge(trans.groupby(['UID'])[feature].count().reset_index(),on='UID',how='left')
#            label =label.merge(trans.groupby(['UID'])[feature].nunique().reset_index(),on='UID',how='left')
#            label =label.merge(trans.groupby(['UID'])[feature].max().reset_index(),on='UID',how='left')
#            label =label.merge(trans.groupby(['UID'])[feature].min().reset_index(),on='UID',how='left')
#            label =label.merge(trans.groupby(['UID'])[feature].sum().reset_index(),on='UID',how='left')
#            label =label.merge(trans.groupby(['UID'])[feature].mean().reset_index(),on='UID',how='left')
#            label =label.merge(trans.groupby(['UID'])[feature].std().reset_index(),on='UID',how='left')
#    return label
#
#
#
#train = get_feature(op_train,trans_train,y).fillna(-1)
#test = get_feature(op_test,trans_test,sub).fillna(-1)
#
#train = train.drop(['UID','Tag'],axis = 1).fillna(-1)
#label = y['Tag']
#
#test_id = test['UID']
#test = test.drop(['UID','Tag'],axis = 1).fillna(-1)
#
#
#lgb_model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=100, reg_alpha=3, reg_lambda=5, max_depth=-1,
#    n_estimators=5000, objective='binary', subsample=0.9, colsample_bytree=0.77, subsample_freq=1, learning_rate=0.05,
#    random_state=1000, n_jobs=16, min_child_weight=4, min_child_samples=5, min_split_gain=0)
#skf = StratifiedKFold(n_splits=5, random_state=2018, shuffle=True)
#best_score = []
#
#oof_preds = np.zeros(train.shape[0])
#sub_preds = np.zeros(test_id.shape[0])
#
#for index, (train_index, test_index) in enumerate(skf.split(train, label)):
#    lgb_model.fit(train.iloc[train_index], label.iloc[train_index], verbose=50,
#                  eval_set=[(train.iloc[train_index], label.iloc[train_index]),
#                            (train.iloc[test_index], label.iloc[test_index])], early_stopping_rounds=30)
#    best_score.append(lgb_model.best_score_['valid_1']['binary_logloss'])
#    print(best_score)
#    oof_preds[test_index] = lgb_model.predict_proba(train.iloc[test_index], num_iteration=lgb_model.best_iteration_)[:,1]
#
#    test_pred = lgb_model.predict_proba(test, num_iteration=lgb_model.best_iteration_)[:, 1]
#    sub_preds += test_pred / 5
#    #print('test mean:', test_pred.mean())
#    #predict_result['predicted_score'] = predict_result['predicted_score'] + test_pred
#
#m = tpr_weight_funtion(y_predict=oof_preds,y_true=label)
#print(m[1])
#sub = pd.read_csv('input/sub.csv')
#sub['Tag'] = sub_preds
#sub.to_csv('sub/baseline_%s.csv'%str(m),index=False)