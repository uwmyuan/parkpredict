# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 15:07:19 2017

@author: yuanyun
"""
#%%input
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns
import numpy as np
import scipy as sp
from datetime import timedelta
import datetime
import platform
#df=ticket user
if (platform.node()=='YundeMacBook-Pro.local'):
    df=pd.read_csv('./LTR.csv')
else:
    df=pd.read_csv('D:\\LTR.csv')
#df.head()
df.apply(lambda x: pd.to_numeric(x, errors='ignore'))
df[['Entr_Time','Exit_Time']]=df[['Entr_Time','Exit_Time']].apply(pd.to_datetime)
df.dtypes
#count
df_entr=df[['Rate','Entr_Time']].set_index('Entr_Time')
df_exit=df[['Rate','Exit_Time']].set_index('Exit_Time')
#plt.figure(1)
##week plot
#plt.plot(df_entr.resample('M').apply(sum))
##7170
##week plot
#plt.plot(df_entr.resample('W').apply(sum))
##2076
##day plot
#plt.plot(df_entr.resample('D').apply(sum))
##2017-04-10  422.0
#plt.title('ticket user')

#plt.figure(2)
##dist one day
day='2017-04-14'
df_entr_pk=df_entr[day]
df_exit_pk=df_exit[day]
#sns.kdeplot(np.array([(x-datetime.datetime(2017,4,10)).total_seconds() for x in df_entr_pk.index.tolist()]))

##dist one week
#plt.figure(3)
#for i in range(7):
#    sns.kdeplot(np.array([(x-datetime.datetime(2017,4,10)).total_seconds() for x in df_entr[str((datetime.datetime(2017,4,10)+timedelta(days=i)).date())].index.tolist()]),label=str((datetime.datetime(2017,4,10)+timedelta(days=i)).date()))
#plt.savefig('D:\\tr_dist_2017-04-10.jpg',dpi=600)
#plt.title('ticket user')
#
#plt.figure(4)
#for i in range(7):
#    sns.kdeplot(np.array([(x-datetime.datetime(2017,4,17)).total_seconds() for x in df_entr[str((datetime.datetime(2017,4,17)+timedelta(days=i)).date())].index.tolist()]),label=str((datetime.datetime(2017,4,17)+timedelta(days=i)).date()))
#plt.savefig('D:\\tr_dist_2017-04-17.jpg',dpi=600)
#plt.title('ticket user')
#
#plt.figure(5)
#for i in range(7):
#    sns.kdeplot(np.array([(x-datetime.datetime(2017,4,17)).total_seconds() for x in df_entr[str((datetime.datetime(2017,4,24)+timedelta(days=i)).date())].index.tolist()]),label=str((datetime.datetime(2017,4,24)+timedelta(days=i)).date()))
#plt.savefig('D:\\ltr_dist_2017-04-24.jpg',dpi=600)
#plt.title('ticket user')

#df2=permit user
if (platform.node()=='YundeMacBook-Pro.local'):
    df2=pd.read_csv('./LPA.csv')
else:
    df2=pd.read_csv('D:\\LPA.csv')
#df2.head()
df2[['Date and Time']]=df2[['Date and Time']].apply(pd.to_datetime)
df2_entr=df2[['Date and Time','Lot']][(df2.Direction=='In')&(df2.Allowed=='Yes')]
#df2_entr.head()
df2_exit=df2[['Date and Time','Lot']][(df2.Direction=='Out')&(df2.Allowed=='Yes')]
#df2_exit.head()
df2_entr=df2_entr[['Date and Time','Lot']].set_index('Date and Time')
df2_exit=df2_exit[['Date and Time','Lot']].set_index('Date and Time')
#dist one day
df2_entr_pk=df2_entr[day]
df2_exit_pk=df2_exit[day]

##%%entering
###count
##df2_entr=df2_entr.set_index('Date and Time')
##
##plt.figure(6)
###week plot
##plt.plot(df_entr.resample('M',how='count'))
###7170
##
###week plot
##plt.figure(7)
##plt.plot(df_entr.resample('W',how='count'))
###2076
##
###day plot
##plt.figure(8)
##plt.plot(df_entr.resample('D',how='count'))
##plt.title('permit user')

plt.figure(9,figsize=(16,8))
sns.kdeplot(np.array([(x-datetime.datetime(2017,4,10)).total_seconds() for x in df2_entr_pk.index.tolist()]),label='permit user arriving',c='r')
sns.kdeplot(np.array([(x-datetime.datetime(2017,4,10)).total_seconds() for x in df_entr_pk.index.tolist()]),label='ticket user arriving',c='b')
plt.title('enter dist one day '+day)
plt.legend()
#plt.savefig('D:\\entr_dist_'+day+'.jpg',dpi=600)


plt.figure(10,figsize=(16,8))
plt.plot([x for x in df_entr_pk.resample('15T',how='count')['Rate'].apply(pd.to_numeric)],label='ticket user arriving',c='b')
plt.plot([x for x in df2_entr_pk.resample('15T',how='count')['Lot'].apply(pd.to_numeric)],label='permit user arriving',c='r')
plt.plot([x+y for x,y in zip(df_entr_pk.resample('15T',how='count')['Rate'].apply(pd.to_numeric),df2_entr_pk.resample('15T',how='count')['Lot'].apply(pd.to_numeric))],label='total arriving',c='g')
plt.title('enter count one day '+day)
plt.legend()
#plt.savefig('D:\\count_'+day+'.jpg',dpi=600)

#dist one week
plt.figure(11,figsize=(16,8))
for i in range(7):
    sns.kdeplot(np.array([(x-datetime.datetime(2017,4,10)).total_seconds() for x in df_entr[str((datetime.datetime(2017,4,10)+timedelta(days=i)).date())].index.tolist()]),label=str((datetime.datetime(2017,4,10)+timedelta(days=i)).date()),c='b')
    sns.kdeplot(np.array([(x-datetime.datetime(2017,4,10)).total_seconds() for x in df2_entr[str((datetime.datetime(2017,4,10)+timedelta(days=i)).date())].index.tolist()]),label=str((datetime.datetime(2017,4,10)+timedelta(days=i)).date()),c='r')
plt.title('enter dist one week')


#%%kstest
#for i in range(1,10):
#    print(sp.stats.kstest(df_entr_pk.resample('15T').count(), 'poisson',[i]))

#print('permit user descriptive stat')
#sp.stats.describe(df2_entr_pk.resample('15T').count())

#%%exiting
plt.figure(9,figsize=(16,8))
sns.kdeplot(np.array([(x-datetime.datetime(2017,4,10)).total_seconds() for x in df2_exit_pk.index.tolist()]),label='permit user exiting',c='r',linestyle='--')
sns.kdeplot(np.array([(x-datetime.datetime(2017,4,10)).total_seconds() for x in df_exit_pk.index.tolist()]),label='ticket user exiting',c='b',linestyle='--')
plt.title('entering & exiting distribution on '+day)
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.legend()
#plt.savefig('D:\\entr_exit_dist_'+day+'.jpg',dpi=600)


plt.figure(12,figsize=(16,8))
plt.plot([x for x in df_exit_pk.resample('15T',how='count')['Rate'].apply(pd.to_numeric)],label='ticket user exiting',c='b',linestyle='--')
plt.plot([x for x in df2_exit_pk.resample('15T',how='count')['Lot'].apply(pd.to_numeric)],label='permit user exiting',c='r',linestyle='--')
plt.plot([x+y for x,y in zip(df_exit_pk.resample('15T',how='count')['Rate'].apply(pd.to_numeric),df2_exit_pk.resample('15T',how='count')['Lot'].apply(pd.to_numeric))],label='total',c='g',linestyle='--')
plt.title('entering & exiting distribution on '+day)
plt.xlabel('Time')
plt.ylabel('Count')
plt.legend()
#plt.savefig('D:\\exit_count_'+day+'.jpg',dpi=600)

#dist one week
plt.figure(13,figsize=(16,8))
for i in range(7):
    sns.kdeplot(np.array([(x-datetime.datetime(2017,4,10)).total_seconds() for x in df_exit[str((datetime.datetime(2017,4,10)+timedelta(days=i)).date())].index.tolist()]),label=str((datetime.datetime(2017,4,10)+timedelta(days=i)).date()),c='b',linestyle='--')
    sns.kdeplot(np.array([(x-datetime.datetime(2017,4,10)).total_seconds() for x in df2_exit[str((datetime.datetime(2017,4,10)+timedelta(days=i)).date())].index.tolist()]),label=str((datetime.datetime(2017,4,10)+timedelta(days=i)).date()),c='r',linestyle='--')
plt.title('entering & exiting distribution on weee starting from '+day)
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.legend(bbox_to_anchor=(1,1))
#plt.savefig('D:\\exit_week_'+day+'.jpg',dpi=600)
#%%dist test
#print('ticket user descriptive stat')
#sp.stats.describe(df_exit_pk.resample('15T',how='count'))
#for i in range(1,10):
#    print(sp.stats.kstest(df_exit_pk.resample('15T').count(), 'poisson',[i]))
#
#print('permit user descriptive stat')
#sp.stats.describe(df2_exit_pk.resample('15T').count())
