#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd


# In[6]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[7]:


import os


# In[8]:


training =pd.read_csv('C:/Users/ADMIN/Desktop/train.csv')
test = pd.read_csv('C:/Users/ADMIN/Desktop/test.csv')
training.head(1)



# In[9]:


test.head(1)


# In[10]:


training['train_test'] = 1
test['train_test']=0
test['Survived']=np.NaN
all_data = pd.concat([training,test])
all_data.head(1)
get_ipython().run_line_magic('matplotlib', 'inline')
all_data.columns


# In[11]:


all_data.head(1)


# In[12]:


training.info()


# In[13]:


training.describe()


# In[14]:


training.describe().columns


# In[15]:


df_num = training[['Age','SibSp','Parch','Fare']]
df_cat = training[['Survived','Pclass','Sex','Ticket','Cabin','Embarked']]


# In[16]:


for i in df_num.columns:
    plt.hist(df_num[i])
    plt.title(i)
    plt.show()


# In[17]:


print(df_num.corr())
sns.heatmap(df_num.corr())


# In[18]:


pd.pivot_table(training,index = 'Survived',values = ['Age','SibSp','Parch','Fare'])


# In[19]:


for i in df_cat.columns:
    sns.barplot(df_cat[i].value_counts().index)
    plt.title(i)
    plt.show()


# In[20]:


print(pd.pivot_table(training,index='Survived',columns='Pclass',values='Ticket',aggfunc='count'))


# In[21]:


print(pd.pivot_table(training,index='Survived',columns='Sex',values='Ticket',aggfunc='count'))


# In[22]:


print(pd.pivot_table(training, index = 'Survived', columns = 'Embarked', values = 'Ticket' ,aggfunc ='count'))


# In[23]:


training['cabin_multiple'] = training.Cabin.apply(lambda x : 0 if pd.isna(x) else len(x.split(' ')))
training['cabin_multiple'].value_counts()


# In[24]:


print(pd.pivot_table(training,index='Survived',columns='cabin_multiple',values='Ticket',aggfunc='count'))


# In[25]:


training['cabin_adv']=training.Cabin.apply(lambda x: str(x)[0])
training.head()


# In[26]:


print(training.cabin_adv.value_counts())


# In[27]:


print(pd.pivot_table(training,index='Survived',columns='cabin_adv',values='Ticket',aggfunc='count',margins='True'))


# In[28]:


training['numeric_ticket']=training.Ticket.apply(lambda x:1 if x.isnumeric() else 0)


# In[29]:


training['ticket_letters'] = training.Ticket.apply(lambda x:''.join(x.split(' ')[:-1]).replace('.','')
.replace('/','').lower() 
if len(x.split(' ')[:-1]) > 0 else 0)
training.ticket_letters


# In[30]:


training['numeric_ticket'].value_counts()


# In[31]:


pd.pivot_table(training,index='Survived',columns='numeric_ticket', values = 'Ticket', aggfunc='count')


# In[32]:


pd.pivot_table(training,index='Survived',columns='ticket_letters', values = 'Ticket', aggfunc='count')


# In[33]:


training['name_title']=training.Name.apply(lambda x:x.split(',')[-1].split('.')[0].strip())
training.head(5)


# In[34]:


training['name_title'].value_counts()


# In[35]:


print(pd.pivot_table(training,index='Survived',columns='name_title',values='Name',aggfunc='count'))


# In[36]:


all_data['cabin_multiple'] = all_data.Cabin.apply(lambda x:0 if pd.isna(x) else len(x.split(' ')))


# In[37]:


all_data['cabin_adv'] = all_data.Cabin.apply(lambda x: str(x)[0])


# In[38]:


all_data['numeric_ticket'] = all_data.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)


# In[39]:


all_data['ticket_letters'] = all_data.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')[:-1]) >0 else 0)


# In[40]:


all_data['name_title'] = all_data.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())


# In[41]:


all_data.Age = all_data.Age.fillna(training.Age.median())


# In[42]:


all_data.Fare = all_data.Fare.fillna(training.Fare.median())


# In[43]:


all_data.dropna(subset=['Embarked'],inplace = True)


# In[44]:


all_data['norm_fare'] = np.log(all_data.Fare+1)
all_data['norm_fare'].hist()
all_data.head()


# In[45]:


all_data['norm_sibsp'] = np.log(all_data.SibSp+1)
all_data['norm_sibsp'].hist()


# In[46]:


all_data.Pclass = all_data.Pclass.astype(str)


# In[47]:


all_dummies = pd.get_dummies(all_data[['Pclass','Sex','Age','SibSp','Parch','norm_fare','Embarked','cabin_adv','cabin_multiple','numeric_ticket','name_title','train_test']])
all_dummies.info()


# In[48]:


all_data.head(1)


# In[ ]:





# In[49]:


X_train = all_dummies[all_dummies.train_test == 1].drop(['train_test'], axis =1)
X_test = all_dummies[all_dummies.train_test == 0].drop(['train_test'], axis =1)
X_test.info()



# In[50]:


y_train = all_data[all_data.train_test==1].Survived
y_train


# In[51]:


all_data.describe().Survived


# In[52]:


training['Survived']


# In[53]:


y_train.size


# In[54]:


from sklearn.preprocessing import StandardScaler
scale = StandardScaler()


# In[55]:


all_dummies_scaled = all_dummies.copy()
all_dummies_scaled.head(2)


# In[56]:


all_dummies_scaled[['Age','SibSp','Parch','norm_fare']]= scale.fit_transform(all_dummies_scaled[['Age','SibSp','Parch','norm_fare']])
all_dummies_scaled.head(2)


# In[57]:


X_train_scaled = all_dummies_scaled[all_dummies_scaled.train_test == 1].drop(['train_test'], axis =1)
X_test_scaled = all_dummies_scaled[all_dummies_scaled.train_test == 0].drop(['train_test'], axis =1)

y_train = all_data[all_data.train_test==1].Survived
y_train.info()


# In[58]:


X_train_scaled.info()


# In[59]:


from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# In[60]:


gnb = GaussianNB()
cv = cross_val_score(gnb,X_train_scaled,y_train,cv=5)
print(cv)
print(cv.mean())


# In[61]:


lr = LogisticRegression(max_iter = 2000)
cv = cross_val_score(lr,X_train,y_train,cv=5)
print(cv)
print(cv.mean())


# In[62]:


dt = tree.DecisionTreeClassifier(random_state = 1)
cv = cross_val_score(dt,X_train,y_train,cv=5)
print(cv)
print(cv.mean())


# In[63]:


dt = tree.DecisionTreeClassifier(random_state = 1)
cv = cross_val_score(dt,X_train_scaled,y_train,cv=5)
print(cv)
print(cv.mean())


# In[64]:


knn = KNeighborsClassifier()
cv = cross_val_score(knn,X_train,y_train,cv=5)
print(cv)
print(cv.mean())


# In[65]:


knn = KNeighborsClassifier()
cv = cross_val_score(knn,X_train_scaled,y_train,cv=5)
print(cv)
print(cv.mean())


# In[66]:


rf = RandomForestClassifier(random_state = 1)
cv = cross_val_score(rf,X_train,y_train,cv=5)
print(cv)
print(cv.mean())


# In[67]:


rf = RandomForestClassifier(random_state = 1)
cv = cross_val_score(rf,X_train_scaled,y_train,cv=5)
print(cv)
print(cv.mean())


# In[68]:


svc = SVC(probability = True)
cv = cross_val_score(svc,X_train_scaled,y_train,cv=5)
print(cv)
print(cv.mean())


# In[69]:


from sklearn.model_selection import GridSearchCV
def clf_performance(classifier, model_name):
    print(model_name)
    
    print('Best Score: ' + str(classifier.best_score_))
    print('Best Parameters: ' + str(classifier.best_params_))
    
lr = LogisticRegression()
param_grid = {'max_iter' : [2000],
              'penalty' : ['l1', 'l2'],
              'C' : np.logspace(-4, 4, 20),
              'solver' : ['liblinear']}

clf_lr = GridSearchCV(lr, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_lr = clf_lr.fit(X_train_scaled,y_train)
clf_performance(best_clf_lr,'Logistic Regression')
clf_performance




# In[70]:


knn = KNeighborsClassifier()
param_grid = {'n_neighbors' : [3,5,7,9],
              'weights' : ['uniform', 'distance'],
              'algorithm' : ['auto', 'ball_tree','kd_tree'],
              'p' : [1,2]}
clf_knn = GridSearchCV(knn, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_knn = clf_knn.fit(X_train_scaled,y_train)
clf_performance(best_clf_knn,'KNN')


# In[71]:


svc = SVC(probability = True)
param_grid = tuned_parameters =[{'kernel': ['rbf'], 'gamma': [.1,.5,1,2,5,10],
                                  'C': [.1, 1, 10, 100, 1000]},
                                 {'kernel': ['linear'], 'C': [.1, 1, 10, 100, 1000]},
                                 {'kernel': ['poly'], 'degree' : [2,3,4,5], 'C': [.1, 1, 10, 100, 1000]}]
clf_svc = GridSearchCV(svc, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_svc = clf_svc.fit(X_train_scaled,y_train)
clf_performance(best_clf_svc,'SVC') 


# In[72]:


rf = RandomForestClassifier(random_state = 1)
param_grid =  {'n_estimators': [400,450,500,550],
               'criterion':['gini','entropy'],
                                  'bootstrap': [True],
                                  'max_depth': [15, 20, 25],
                                  'max_features': ['auto','sqrt', 10],
                                  'min_samples_leaf': [2,3],
                                  'min_samples_split': [2,3]}
                                  
clf_rf = GridSearchCV(rf, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_rf = clf_rf.fit(X_train_scaled,y_train)
clf_performance(best_clf_rf,'Random Forest')


# In[73]:


best_rf = best_clf_rf.best_estimator_.fit(X_train_scaled,y_train)
feat_importances = pd.Series(best_rf.feature_importances_, index=X_train_scaled.columns)
feat_importances.nlargest(20).plot(kind='barh')


# In[76]:


import xgboost
from xgboost import XGBClassifier
xgb = XGBClassifier(random_state = 1)

param_grid = {
    'n_estimators': [450,500,550],
    'colsample_bytree': [0.75,0.8,0.85],
    'max_depth': [None],
    'reg_alpha': [1],
    'reg_lambda': [2, 5, 10],
    'subsample': [0.55, 0.6, .65],
    'learning_rate':[0.5],
    'gamma':[.5,1,2],
    'min_child_weight':[0.01],
    'sampling_method': ['uniform']
}

clf_xgb = GridSearchCV(xgb, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
best_clf_xgb = clf_xgb.fit(X_train_scaled,y_train)
clf_performance(best_clf_xgb,'XGB')


# In[77]:


y_hat_xgb = best_clf_xgb.best_estimator_.predict(X_test_scaled).astype(int)
xgb_submission = {'PassengerId': test.PassengerId, 'Survived': y_hat_xgb}
submission_xgb = pd.DataFrame(data=xgb_submission)
submission_xgb.to_csv('xgb_submission3.csv', index=False)


# In[80]:


from sklearn.ensemble import RandomForestClassifier, VotingClassifier
best_lr = best_clf_lr.best_estimator_
best_knn = best_clf_knn.best_estimator_
best_svc = best_clf_svc.best_estimator_
best_rf = best_clf_rf.best_estimator_
best_xgb = best_clf_xgb.best_estimator_

voting_clf_hard = VotingClassifier(estimators = [('knn',best_knn),('rf',best_rf),('svc',best_svc)], voting = 'hard') 
voting_clf_soft = VotingClassifier(estimators = [('knn',best_knn),('rf',best_rf),('svc',best_svc)], voting = 'soft') 
voting_clf_all = VotingClassifier(estimators = [('knn',best_knn),('rf',best_rf),('svc',best_svc), ('lr', best_lr)], voting = 'soft') 
voting_clf_xgb = VotingClassifier(estimators = [('knn',best_knn),('rf',best_rf),('svc',best_svc), ('xgb', best_xgb),('lr', best_lr)], voting = 'soft')

print('voting_clf_hard :',cross_val_score(voting_clf_hard,X_train,y_train,cv=5))
print('voting_clf_hard mean :',cross_val_score(voting_clf_hard,X_train,y_train,cv=5).mean())

print('voting_clf_soft :',cross_val_score(voting_clf_soft,X_train,y_train,cv=5))
print('voting_clf_soft mean :',cross_val_score(voting_clf_soft,X_train,y_train,cv=5).mean())

print('voting_clf_all :',cross_val_score(voting_clf_all,X_train,y_train,cv=5))
print('voting_clf_all mean :',cross_val_score(voting_clf_all,X_train,y_train,cv=5).mean())

print('voting_clf_xgb :',cross_val_score(voting_clf_xgb,X_train,y_train,cv=5))
print('voting_clf_xgb mean :',cross_val_score(voting_clf_xgb,X_train,y_train,cv=5).mean())


# In[81]:


params = {'weights' : [[1,1,1],[1,2,1],[1,1,2],[2,1,1],[2,2,1],[1,2,2],[2,1,2]]}

vote_weight = GridSearchCV(voting_clf_soft, param_grid = params, cv = 5, verbose = True, n_jobs = -1)
best_clf_weight = vote_weight.fit(X_train_scaled,y_train)
clf_performance(best_clf_weight,'VC Weights')
voting_clf_sub = best_clf_weight.best_estimator_.predict(X_test_scaled)


# In[82]:


voting_clf_hard.fit(X_train_scaled, y_train)
voting_clf_soft.fit(X_train_scaled, y_train)
voting_clf_all.fit(X_train_scaled, y_train)
voting_clf_xgb.fit(X_train_scaled, y_train)

best_rf.fit(X_train_scaled, y_train)
y_hat_vc_hard = voting_clf_hard.predict(X_test_scaled).astype(int)
y_hat_rf = best_rf.predict(X_test_scaled).astype(int)
y_hat_vc_soft =  voting_clf_soft.predict(X_test_scaled).astype(int)
y_hat_vc_all = voting_clf_all.predict(X_test_scaled).astype(int)
y_hat_vc_xgb = voting_clf_xgb.predict(X_test_scaled).astype(int)


# In[83]:


final_data = {'PassengerId': test.PassengerId, 'Survived': y_hat_rf}
submission = pd.DataFrame(data=final_data)

final_data_2 = {'PassengerId': test.PassengerId, 'Survived': y_hat_vc_hard}
submission_2 = pd.DataFrame(data=final_data_2)

final_data_3 = {'PassengerId': test.PassengerId, 'Survived': y_hat_vc_soft}
submission_3 = pd.DataFrame(data=final_data_3)

final_data_4 = {'PassengerId': test.PassengerId, 'Survived': y_hat_vc_all}
submission_4 = pd.DataFrame(data=final_data_4)

final_data_5 = {'PassengerId': test.PassengerId, 'Survived': y_hat_vc_xgb}
submission_5 = pd.DataFrame(data=final_data_5)

final_data_comp = {'PassengerId': test.PassengerId, 'Survived_vc_hard': y_hat_vc_hard, 'Survived_rf': y_hat_rf, 'Survived_vc_soft' : y_hat_vc_soft, 'Survived_vc_all' : y_hat_vc_all,  'Survived_vc_xgb' : y_hat_vc_xgb}
comparison = pd.DataFrame(data=final_data_comp)


# In[84]:


comparison['difference_rf_vc_hard'] = comparison.apply(lambda x: 1 if x.Survived_vc_hard != x.Survived_rf else 0, axis =1)
comparison['difference_soft_hard'] = comparison.apply(lambda x: 1 if x.Survived_vc_hard != x.Survived_vc_soft else 0, axis =1)
comparison['difference_hard_all'] = comparison.apply(lambda x: 1 if x.Survived_vc_all != x.Survived_vc_hard else 0, axis =1)


# In[85]:


comparison.difference_hard_all.value_counts()


# In[89]:


submission.to_csv('C:/Users/ADMIN/Desktop/Taitanic/submission_rf.csv', index =False)
submission_2.to_csv('C:/Users/ADMIN/Desktop/Taitanic/submission_vc_hard.csv',index=False)
submission_3.to_csv('C:/Users/ADMIN/Desktop/Taitanic/submission_vc_soft.csv', index=False)
submission_4.to_csv('C:/Users/ADMIN/Desktop/Taitanic/submission_vc_all.csv', index=False)
submission_5.to_csv('C:/Users/ADMIN/Desktop/Taitanic/submission_vc_xgb2.csv', index=False)


# In[ ]:




