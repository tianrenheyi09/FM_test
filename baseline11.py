# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 20:33:50 2018

@author: Administrator
"""
import time
import datetime
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


def get_time_unix(data,month):
#    data['month'] = data.month.apply(lambda x:str(x))
    data['day'] = data.day.apply(lambda x:str(x))
    data['time_stamp'] = '2018-'+str(month)+'-'+data['day']+' '+data['time']
    data['time_stamp'] = data.time_stamp.apply(lambda x:datetime.datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))
    data['time_stamp'] = data.time_stamp.apply(lambda x:int(time.mktime(x.timetuple())))
    return data[['UID','time_stamp']]


#########查看每个特征的缺失个数
null_column=pd.DataFrame()
null_column['trans_train']=trans_train.isnull().sum()/len(trans_train['UID'])
null_column['trans_test']=trans_test.isnull().sum()/len(trans_test['UID'])
null_column

op_null_column = pd.DataFrame()
op_null_column ['op_train']=op_train.isnull().sum()/len(op_train['UID'])
op_null_column ['op_test']=op_test.isnull().sum()/len(op_test['UID'])
op_null_column
#
#op_train =pd.concat([op_train,op_test],axis=0,ignore_index=True)
#trans_train = pd.concat([trans_train,trans_test],axis=0,ignore_index=True)

#def get_row_null(data,vname):
#    data[vname+'lv_row_null']=data.isnull().sum(axis=1)/len(data.columns.tolist())
#    data[vname+'row_null']=data.isnull().sum(axis=1)
#    return data
#
#op_train = get_row_null(op_train,'op_train')
#trans_train = get_row_null(trans_train,'trans_train')


import scipy.stats as sp
from collections import Counter


def get_time_gap(strs,param):
    time = strs.split(":")
    time = list(set(time))
    time = sorted(list(map(lambda x:int(x),time)))
    time_gap = []
    if(len(time)==1):
        return -20
    for index,value in enumerate(time):
        if(index<=len(time)-2):
            gap = abs(time[index]-time[index+1])
            time_gap.append(gap)
    
    if param=='1':
        return np.mean(time_gap)
    elif param=='2':
        return np.max(time_gap)
    elif param=='3':
        return np.min(time_gap)
    elif param=='4':
        return np.std(time_gap)
    elif param=='5':
        return sp.stats.skew(time_gap)
    elif param=='6':
        return sp.stats.kurtosis(time_gap)

def get_continue_launch_count(strs,param):
    time = strs.split(":")
    time = dict(Counter(time))
    time = sorted(time.items(),key=lambda x:x[0],reverse=False)
    key_list = []
    value_list = []
    if(len(time)==1):
        return -2
    for key,value in dict(time).items():
        key_list.append(int(key))
        value_list.append(int(value))
    
    if np.mean(np.diff(key_list,1))==1:
        if param=='1':
            return np.mean(value_list)
        elif param=='2':
            return np.max(value_list)
        elif param=='3':
            return np.min(value_list)
        elif param=='4':
            return np.std(value_list)
    else:
        return -1

def cur_day_repeat_count(strs):
    time = strs.split(":")
    time = dict(Counter(time))
    time = sorted(time.items(),key=lambda x:x[1],reverse=False)
    #一天一次启动
    if(len(time)==1)&(time[0][1]==1):
        return 0
    #一天多次
    elif (len(time)==1)&(time[0][1]>1):
        return 1
    ###多天多次
    elif (len(time)>1)&(time[0][1]>=2):
        return 2
    else:
        return 3

def get_lianxu_day(day_list):
    time = day_list.split(":")
    time = list(map(lambda x:int(x),time))
    m = np.array(time)
    if(len(set(m))==1):
        return -1
    m = list(set(m))
    if(len(m)==0):
        return -20
    n = np.where(np.diff(m)==1)[0]
    i = 0
    result = []
    while i<len(n)-1:
        state = 1
        while n[i+1]-n[i]==1:
            state +=1
            i +=1
            if i==len(n)-1:
                break
        if state==1:
            i+=1
            result.append(2)
        else:
            i+=1
            result.append(state+1)
    if len(n)==1:
        result.append(2)
    if len(result)!=0:
        return np.max(result)

def get_week(day):
    day = int(day)
    if day>=1 and day<=7:
        return 1
    if day>=8 and day<=14:
        return 2
    if day>=15 and day<=21:
        return 3
    if day>=22 and day<=28:
        return 4
    if day>=28:
        return 5

data = trans_train.copy()
data = data.sort_values(by='day')
data['hour'] = data['time'].apply(lambda x:int(x[0:2]))
data['hour'] = data['hour'].astype('str')
tfea1 = data[['UID','hour']]
tfea1 = tfea1.groupby(['UID'])['hour'].agg(lambda x:':'.join(x)).reset_index()

###############用户操作次数
def process_fea(data1,vname):
    data = data1.copy()
    
    data['week'] = data['day'].apply(get_week)
    data['hour'] = data['time'].apply(lambda x:int(x[0:2]))
    
    feat3 = data[['UID','day']]
    feat3['day'] = feat3['day'].astype('str')
    feat3 = feat3.groupby(['UID'])['day'].agg(lambda x:':'.join(x)).reset_index()
    feat3.rename(columns={'day':'act_list'},inplace=True)
    ###用户是否多天有多次启动（均值）
    feat3['time_gap_mean']=feat3['act_list'].apply(get_time_gap,args=('1'))
    ###最大
    feat3['time_gap_max']=feat3['act_list'].apply(get_time_gap,args=('2'))
    ###最小
    feat3['time_gap_min']=feat3['act_list'].apply(get_time_gap,args=('3'))
    ####方差
    feat3['time_gap_std']=feat3['act_list'].apply(get_time_gap,args=('4'))
    ###峰度
    feat3['time_gap_skew']=feat3['act_list'].apply(get_time_gap,args=('5'))
    ###偏度
    feat3['time_gap_kurt']=feat3['act_list'].apply(get_time_gap,args=('6'))
    ###平均行为次数
    feat3['mean_act_count'] = feat3['act_list'].apply(lambda x:len(x.split(":"))/(len(set(x.split(":")))))
    ###平均行为日期
    feat3['act_mean_date'] = feat3['act_list'].apply(lambda x:np.sum([int(ele) for ele in x.split(":")])/len(x.split(":")))
    ###活动天数占一个月的比例
    feat3['act_rate'] = feat3['act_list'].apply(lambda x:len(list(set(x.split(":"))))/31)
    ###用户是否当天有多次启动
    feat3['cur_day_repeat_count'] = feat3['act_list'].apply(cur_day_repeat_count)
    ###连续几天启动次数的均值
    feat3['con_act_day_count_mean'] = feat3['act_list'].apply(get_continue_launch_count,args=('1'))
    feat3['con_act_day_count_max'] = feat3['act_list'].apply(get_continue_launch_count,args=('2'))
    feat3['con_act_day_count_min'] = feat3['act_list'].apply(get_continue_launch_count,args=('3'))
    feat3['con_act_day_count_total'] = feat3['act_list'].apply(get_continue_launch_count,args=('4'))
    feat3['con_act_day_count_std'] = feat3['act_list'].apply(get_continue_launch_count,args=('5'))
    feat3['con_act_max'] = feat3['act_list'].apply(get_lianxu_day)
    del feat3['act_list']
    
    ###判断日期是否为周末
    high_act_day_list = [7,14,21,28]
    feat8 = data[['UID','day']]
    feat8['is_high_act'] = feat8['day'].apply(lambda x:1 if x in high_act_day_list else 0)
    feat8 = feat8.drop_duplicates(subset=['UID'])
    del feat8['day']
    
    ####
    feat10 = data.groupby(['UID','day'],as_index=False)['time'].agg({'user_per_count': "count"})
    ###用户平均每天启动次数
    feat10_copy = feat10.copy()
    feat11 = feat10_copy.groupby(['UID'],as_index=False)['user_per_count'].agg({"user_per_mean":"mean"})
    ####每天启动次数最大值
    feat12 = feat10_copy.groupby(['UID'],as_index=False)['user_per_count'].agg({"user_per_max":"max"})
    feat13 = feat10_copy.groupby(['UID'],as_index=False)['user_per_count'].agg({"user_per_min":"min"})
    feat14 = feat10_copy.groupby(['UID'],as_index=False)['user_per_count'].agg({"user_modle_count":lambda x:x.value_counts().index[0]})
    feat15 = feat10_copy.groupby(['UID'],as_index=False)['user_per_count'].agg({"user_per_std":np.std})
    feat16 = feat10_copy.groupby(['UID'],as_index=False)['user_per_count'].agg({"user_per_skew":sp.stats.skew})
    feat17 = feat10_copy.groupby(['UID'],as_index=False)['user_per_count'].agg({"user_per_kurt":sp.stats.kurtosis})
    feat18 = feat10_copy.groupby(['UID'],as_index=False)['user_per_count'].agg({"user_per_median":np.median})
    
    ########时间time特征
    if(vname=='op'):
        feat27 = get_time_unix(data,7)
        print("fea27 is op")
    if(vname=='trans'):
        feat27 = get_time_unix(data,8)
        print("feat 27 is trans")
    
    log =feat27.sort_values(['UID','time_stamp'])
    log['next_time'] = log.groupby(['UID'])['time_stamp'].diff(-1)
    log['before_time'] = log.groupby(['UID'])['time_stamp'].diff(1)
    log_next = log.groupby(['UID'],as_index=False)['next_time'].agg(
    {"next_time_mean":np.mean,
     "next_time_std":np.std,
     "next_time_min":np.min,
     "next_time_max":np.max})
    
    log_before = log.groupby(['UID'],as_index=False)['before_time'].agg(
    {"before_time_mean":np.mean,
     "before_time_std":np.std,
     "begore_time_min":np.min,
     "before_time_max":np.max})
    
    ##########每周的统计特征
    feat_week = data.groupby(['UID','week'],as_index=False)['day'].agg({"user_week_count":"count"})
    week_sp = feat_week.copy()
    feat11_sp = week_sp.groupby(['UID'],as_index=False)['user_week_count'].agg({"user_week_max":"max"})
    feat12_sp =  week_sp.groupby(['UID'],as_index=False)['user_week_count'].agg({"user_week_min":"min"})
    feat13_sp =  week_sp.groupby(['UID'],as_index=False)['user_week_count'].agg({"user_week_mean":"mean"})
    feat14_sp =  week_sp.groupby(['UID'],as_index=False)['user_week_count'].agg({"user_week_mode":lambda x:x.value_counts().index[0]})
    feat15_sp =  week_sp.groupby(['UID'],as_index=False)['user_week_count'].agg({"user_week_std":np.std})
    feat16_sp =  week_sp.groupby(['UID'],as_index=False)['user_week_count'].agg({"user_week_skew":sp.stats.skew})
    feat17_sp =  week_sp.groupby(['UID'],as_index=False)['user_week_count'].agg({"user_week_kurt":sp.stats.kurtosis})
    feat18_sp =  week_sp.groupby(['UID'],as_index=False)['user_week_count'].agg({"user_week_median":np.median})
    
    
    data = data[['UID']]

    data = data.drop_duplicates(subset='UID')
    data = pd.merge(data,feat3,on='UID',how='left')
    data = pd.merge(data,feat8,on='UID',how='left')
    data = pd.merge(data,feat11,on='UID',how='left')
    data = pd.merge(data,feat12,on='UID',how='left')
    data = pd.merge(data,feat13,on='UID',how='left')
    data = pd.merge(data,feat14,on='UID',how='left')
    data = pd.merge(data,feat15,on='UID',how='left')
    data = pd.merge(data,feat16,on='UID',how='left')
    data = pd.merge(data,feat17,on='UID',how='left')
    data = pd.merge(data,feat18,on='UID',how='left')
    data = pd.merge(data,log_next,on='UID',how='left')
    data = pd.merge(data,log_before,on='UID',how='left')
    data = pd.merge(data,feat11_sp,on='UID',how='left')
    data = pd.merge(data,feat12_sp,on='UID',how='left')
    data = pd.merge(data,feat13_sp,on='UID',how='left')
    data = pd.merge(data,feat14_sp,on='UID',how='left')
    data = pd.merge(data,feat15_sp,on='UID',how='left')
    data = pd.merge(data,feat16_sp,on='UID',how='left')
    data = pd.merge(data,feat17_sp,on='UID',how='left')
    data = pd.merge(data,feat18_sp,on='UID',how='left')
    
    d_columns = data.columns.tolist()
    d_columns.remove('UID')

        
    for ff in d_columns:
        if(vname=='op'):
            data=data.rename(columns={ff:'op_'+ff})
        if(vname=='trans'):
            data=data.rename(columns={ff:'trans_'+ff})
        
    
    
    return data

##################删除缺失值太多的特征


#op_columns = op_train.columns.tolist()
#trans_columns = trans_train.columns.tolist()
#for feature in op_null_column.index.tolist():
#    if(op_null_column.ix[feature,'op_train']>=0.5):
#        op_train.drop(feature,axis=1,inplace=True)
#    
#for feature in null_column.index.tolist():
#    if(null_column.ix[feature,'trans_train']>=0.5):
#        trans_train.drop(feature,axis=1,inplace=True)


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


def get_low_fea(data,fea_list,flag):
    
    sub_data = data[['UID']].drop_duplicates(subset='UID')
    for fea in fea_list:
        u = data[['UID',fea]]
        ofea0 = u.groupby(['UID'])[fea].count().reset_index().rename(columns={fea:flag+fea+'_count'})
        u1 = u.groupby(['UID'])[fea].nunique().reset_index().rename(columns={fea:flag+fea+'_nun'})
        ofea0 = pd.merge(ofea0,u1,on='UID',how='left')
        sub_data = pd.merge(sub_data,ofea0,on='UID',how='left')
    
    return sub_data


#u = data2.groupby(['UID'])['bal'].agg({'cbal':lambda x:x.value_counts().index[0]}).reset_index()
#u = data2.groupby(['UID'])['bal'].mean().reset_index()
#feat14 = feat10_copy.groupby(['UID'],as_index=False)['user_per_count'].agg({"user_modle_count":lambda x:x.value_counts().index[0]})

def get_sta_fea(data,fea_list,flag):
    
    sub_data = data[['UID']].drop_duplicates(subset='UID')
    for fea in fea_list:
        u = data.groupby(['UID'])[fea].count().reset_index().rename(columns={fea:flag+fea+'_count'})
        sub_data =pd.merge(sub_data,u,on='UID',how='left')
        u = data.groupby(['UID'])[fea].nunique().reset_index().rename(columns={fea:flag+fea+'_nun'})
        sub_data =pd.merge(sub_data,u,on='UID',how='left')
        u = data.groupby(['UID'])[fea].max().reset_index().rename(columns={fea:flag+fea+'_max'})
        sub_data =pd.merge(sub_data,u,on='UID',how='left')
        u = data.groupby(['UID'])[fea].min().reset_index().rename(columns={fea:flag+fea+'_min'})
        sub_data =pd.merge(sub_data,u,on='UID',how='left')
        u = data.groupby(['UID'])[fea].mean().reset_index().rename(columns={fea:flag+fea+'_mean'})
        sub_data =pd.merge(sub_data,u,on='UID',how='left')
        u = data.groupby(['UID'])[fea].sum().reset_index().rename(columns={fea:flag+fea+'_sum'})
        sub_data =pd.merge(sub_data,u,on='UID',how='left')
        u = data.groupby(['UID'])[fea].std().reset_index().rename(columns={fea:flag+fea+'_std'})
        sub_data =pd.merge(sub_data,u,on='UID',how='left')
        u = data.groupby(['UID'])[fea].agg({flag+fea+'_skew':sp.stats.skew}).reset_index()
        sub_data = pd.merge(sub_data,u,on='UID',how='left')
        u = data.groupby(['UID'])[fea].agg({flag+fea+'_kur':sp.stats.kurtosis}).reset_index()
        sub_data = pd.merge(sub_data,u,on='UID',how='left')
        u = data.groupby(['UID'])[fea].agg({flag+fea+'_median':np.median}).reset_index()
        sub_data = pd.merge(sub_data,u,on='UID',how='left')
        u = data.groupby(['UID'])[fea].agg({flag+fea+'_modz':lambda x:x.value_counts().index[0]}).reset_index()
        sub_data = pd.merge(sub_data,u,on='UID',how='left')
    

    return sub_data

#['trans_amt','bal','total_amt']
#fea='trans_amt'
#flag='trans'
#u = add_trans.groupby(['UID'])[fea].agg({flag+fea+'_modz':lambda x:x.value_counts().index[0]}).reset_index()



def pro_of_op(data1):
    #######device_code   
    u = data1[['device_code1','device_code2','device_code3']]
    u = u.fillna('?')
    u['device_code_op'] = u['device_code1']+u['device_code2']+u['device_code3']
    u['device_code_op'] = u['device_code_op'].replace('???',np.NaN)
    
    data1['device_code_op'] = u['device_code_op']
    ###########  mac
    u = data1[['mac1','mac2']]
    u = u.fillna('?')
    u['mac_op'] = u['mac1']+u['mac2']
    u['mac_op'] = u['mac_op'].replace('??',np.NaN)
    
    data1['mac_op'] = u['mac_op']
    
    ##########  ip
    u = data1[['ip1','ip2']]
    u = u.fillna('?')
    u['ip_op'] = u['ip1']+u['ip2']
    u['ip_op'] = u['ip_op'].replace('??',np.NaN)
    
    data1['ip_op'] = u['ip_op']
    
    ##############  ip_sub
    u = data1[['ip1_sub','ip2_sub']]
    u = u.fillna('?')
    u['ip_sub_op'] = u['ip1_sub']+u['ip2_sub']
    u['ip_sub_op'] = u['ip_sub_op'].replace('??',np.NaN)
    
    data1['ip_sub_op'] =u['ip_sub_op']
    
    return data1

#######  data2  trans_train
def pro_of_trans(data2):
#    fea_trans = pd.DataFrame()
#    fea_trans['value'] = trans_train.columns.tolist()
#    data2 = trans_train.copy()
    
    ######  data2 code
    u = data2[['code1','code2']]
    u = u.fillna('?')
    u['code_trans'] = u['code1']+u['code2']
    u['code_trans'] = u['code_trans'].replace('??',np.NaN)
    
    data2['code_trans'] = u['code_trans']
    #####  data2  ant_src
    u = data2[['amt_src1','amt_src2']]
    u = u.fillna('?')
    u['amt_src_trans'] = u['amt_src1']+u['amt_src2']
    u['amt_src_trans'] = u['amt_src_trans'].replace('??',np.NaN)
    
    data2['amt_src_trans'] = u['amt_src_trans']
    ######## data2  trans_typr
#    u = data2[['trans_type1','trans_type2']]
#    u = u.fillna('?')
#    u['trans_type_tr'] = u['trans_type1']+u['trans_type2']
#    u['trans_type_tr'] = u['trans_type_tr'].replace('??',np.NaN)
#    
#    data2['trans_type_tr'] = u['trans_type_tr']
    ##### data2 acc_id 
    u = data2[['acc_id1','acc_id2','acc_id3']]
    u = u.fillna('?')
    u['acc_trans'] = u['acc_id1']+u['acc_id2']+u['acc_id3']
    u['acc_trans'] = u['acc_trans'].replace('???',np.NaN)
    
    data2['acc_trans'] = u['acc_trans']
    
    ######   data2  device_code
    u = data2[['device_code1','device_code2','device_code3']]
    u = u.fillna('?')
    u['device_code_trans'] = u['device_code1']+u['device_code2']+u['device_code3']
    u['device_code_trans'] = u['device_code_trans'].replace('???',np.NaN)
    
    data2['device_code_trans'] = u['device_code_trans']
    
    ############  data2  device
    u = data2[['device1','device2']]
    u = u.fillna('?')
    u['device_trans'] = u['device1']+u['device2']
    u['device_trans'] = u['device_trans'].replace('??',np.NaN)
    
    data2['device_trans'] = u['device_trans']
    
    u = data2[['trans_amt','bal']]
    u['total_amt'] = u['trans_amt']+u['bal']
    u['amt_lv'] = u['trans_amt']/u['total_amt']
    
    data2['total_amt'] = u['total_amt']
    data2['amt_lv'] = u['amt_lv']
    
    return data2

#########合并操作和交易的看看
u = op_train[['UID','day','time']]
u1 = trans_train[['UID','day','time','trans_amt']]
u = pd.merge(u,u1,on=['UID','day','time'],how='left')
u2 = u[u.UID==10035]


data_train = y[['UID']]
data_test = sub[['UID']]

add_op = pd.concat([op_train,op_test],axis=0,ignore_index=True)
add_trans = pd.concat([trans_train,trans_test],axis=0,ignore_index=True)

add_op = pro_of_op(add_op)
add_trans = pro_of_trans(add_trans)

###提取统计特征
add_op_fea= add_op.columns.tolist()
add_op_fea.remove('UID')
add_op_fea.remove('time')

add_op_fea1 = get_low_fea(add_op,add_op_fea,'op')

add_trans_fea = add_trans.columns.tolist()
add_trans_fea.remove('UID')
add_trans_fea.remove('time')

sta_fea  = ['trans_amt','bal','total_amt']
add_trans_fea = list(set(add_trans_fea).difference(set(sta_fea)))

add_trans_fea1 = get_low_fea(add_trans,add_trans_fea,'trans')

#####提取树脂类的特征
add_trans_fea2 = get_sta_fea(add_trans,sta_fea,'trans')


##训练集的操作的时间特征
#time_train_fea1 = process_fea(op_train,'op')
#time_train_fea2 = process_fea(trans_train,'trans')
#####测试集的操作与交易的时间特征
#time_test_fea1 = process_fea(op_test,'op')
#time_test_fea2 = process_fea(trans_test,'trans')


##########训练集特征矩阵
data_train=pd.merge(data_train,add_op_fea1,on='UID',how='left')
data_train = pd.merge(data_train,add_trans_fea1,on='UID',how='left')
data_train = pd.merge(data_train,add_trans_fea2,on='UID',how='left')

###
#data_train = pd.merge(data_train,time_train_fea1,on='UID',how='left')
#data_train = pd.merge(data_train,time_train_fea2,on='UID',how='left')

#############3测试集特征
data_test = pd.merge(data_test,add_op_fea1,on='UID',how='left')
data_test = pd.merge(data_test,add_trans_fea1,on='UID',how='left')
data_test = pd.merge(data_test,add_trans_fea2,on='UID',how='left')

#data_test =pd.merge(data_test,time_test_fea1,on='UID',how='left')
#data_test = pd.merge(data_test,time_test_fea2,on='UID',how='left')





train = data_train.drop(['UID'],axis=1)
test = data_test.drop(['UID'],axis=1)


label = y['Tag']

test_id = sub['UID']


#lgb_model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=100, reg_alpha=3, reg_lambda=5, max_depth=-1,
#    n_estimators=5000, objective='binary', subsample=0.9, colsample_bytree=0.77, subsample_freq=1, learning_rate=0.05,
#      min_child_weight=4, min_child_samples=5, min_split_gain=0)

lgb_model = lgb.LGBMClassifier(boosting_type='gbdt', 
                               num_leaves=15,
#                               reg_alpha=3, 
#                               reg_lambda=5, 
                               max_depth=-1,
                               n_estimators=10000, 
                               objective='binary', 
                               subsample=0.8, 
                               colsample_bytree=0.7, 
#                               subsample_freq=1, 
                               learning_rate=0.05,
                               seed=2018, 
#                               n_jobs=16, 
#                               min_child_weight=4,
#                               min_child_samples=5, 
#                               min_split_gain=0
                               )


skf = StratifiedKFold(n_splits=5, random_state=2018, shuffle=True)
best_score = []

oof_preds = np.zeros(train.shape[0])
sub_preds = np.zeros(test_id.shape[0])


for index, (train_index, test_index) in enumerate(skf.split(train, label)):
    lgb_model.fit(train.iloc[train_index], label.iloc[train_index], verbose=50,
                  eval_set=[(train.iloc[train_index], label.iloc[train_index]),
                            (train.iloc[test_index], label.iloc[test_index])],early_stopping_rounds=100)
#    best_score.append(lgb_model.best_score_['valid_1']['binary_logloss'])
#    print(best_score)
    oof_preds[test_index] = lgb_model.predict_proba(train.iloc[test_index], num_iteration=lgb_model.best_iteration)[:,1]

    test_pred = lgb_model.predict_proba(test, num_iteration=lgb_model.best_iteration)[:, 1]
    sub_preds += test_pred / 5
    #print('test mean:', test_pred.mean())
    #predict_result['predicted_score'] = predict_result['predicted_score'] + test_pred


predictors = [i for i in train.columns]
feat_imp = pd.Series(lgb_model.feature_importance(), predictors).sort_values(ascending=False)
print(feat_imp)
print(feat_imp.shape)



m = tpr_weight_funtion(y_predict=oof_preds,y_true=label)
print(m[1])
sub = pd.read_csv('sub.csv')
sub['Tag'] = sub_preds
sub.to_csv('sub_baseline_%s.csv'%str(m),index=False)