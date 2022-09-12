# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 13:51:30 2022

@author: Kareem
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 18:10:16 2022

@author: Kareem
"""

import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
from PIL import Image


image=Image.open(r"C:/Users/Kareem/Documents/proj7/personal_loans.jpg")
print(image)
st.set_page_config(
     page_title="Personal loan campaign",
     page_icon='🏦',
     layout="wide",
     initial_sidebar_state="expanded",
      )

st.title(" 🏦 Thera Bank's Personal loan campaign : a Machine learning project ") # set title of dashboard 
col1,col2,col3=st.columns(3)
with col2:
    st.image(image,width=500) 

st.title(" Description")
st.write("Thera bank ran a campaign last year hoping to convert its liability customers into personal loan clients as well.")
st.write(" Using the data they provided on their customers' age, income, education , family and their relationship with the bank ")
st.write("we deploy machine learning to help the bank determine whether a customer is more likely to take a personal loan and shift attention towards those customers who will")
df=pd.read_excel(r'C:/Users/Kareem/Desktop/proj7_ex.xlsx')
plot_df=pd.read_excel(r'C:/Users/Kareem/Desktop/proj7_ex.xlsx')

plot_df.loc[plot_df['Personal Loan']==0, 'Outcome']='No Personal Loan'
plot_df.loc[plot_df['Personal Loan']==1, 'Outcome']='Personal Loan'
plot_df.loc[plot_df['Education']==1, 'Ed level']='Primary'
plot_df.loc[plot_df['Education']==2, 'Ed level']='Secondary'
plot_df.loc[plot_df['Education']==3, 'Ed level']='Tertiary'

df.drop(columns='ID',axis=1,inplace=True)

st.sidebar.title('👇 Predict if a customer will take a personal loan 👇')

st.sidebar.subheader('1- Age')
age_sel=st.sidebar.number_input(" Integers only  ")


st.sidebar.subheader('2- Experience')
exp_sel=st.sidebar.number_input(" Integers only ")


st.sidebar.subheader('3- Income')
inc_sel=st.sidebar.number_input(" in $k/year ")



st.sidebar.subheader('4- Zip code')
zip_sel=st.sidebar.selectbox(label='Select location',options=df['ZIP Code'].unique().tolist())



st.sidebar.subheader('5- Family size')
fam_sel=st.sidebar.number_input(" Family members")


st.sidebar.subheader('6- Credit card spending')
cred_sel=st.sidebar.number_input(" $k/month ")


st.sidebar.subheader('7- Education level')
edu_sel=st.sidebar.selectbox(label='Primary,Secondary or Tertiary',options=['Primary','Secondary','Tertiary'])
if edu_sel=='Primary':
    edu_sel=1
elif edu_sel == 'Secondary':
    edu_sel=2
elif edu_sel == 'Tertiary':
    edu_sel=3


st.sidebar.subheader('8- Mortgage' )
mor_sel=st.sidebar.number_input(" Mortgage duration months ")

st.sidebar.subheader('9- Securities account ' )
#sec_sel=st.sidebar.selectbox(label='Yes or No',options=['Yes ','No'])
sec_sel=st.sidebar.radio(
     "Does the client have a security account?",
     ('Yes', 'No'))
if sec_sel.rstrip()=='Yes':
    sec_sel=1
elif sec_sel.rstrip() == 'No':
    sec_sel=0

st.sidebar.subheader('10- CD account' )
#cd_sel=st.sidebar.selectbox(label='Yes or No',options=['Yes','No '])
cd_sel=st.sidebar.radio(
     "Does the client have a deposit account?",
     ('Yes', 'No'))
if cd_sel.rstrip()=='Yes           ':
    cd_sel=1
elif cd_sel.rstrip() == 'No':
    cd_sel=0

st.sidebar.subheader('11- Online account' )
#on_sel=st.sidebar.selectbox(label='Yes or No',options=['Yes     ','No '])
on_sel=st.sidebar.radio(
     "Does the client have an online account?",
     ('Yes    ', 'No'))
if on_sel.rstrip()=='Yes':
    on_sel=1
elif on_sel.rstrip() == 'No':
    on_sel=0
    
    
st.sidebar.subheader('12- Credit card holder' )
#cred_sel=st.sidebar.selectbox(label='Yes or No',options=['Yes','No      '])
cred_sel=st.sidebar.radio(
     "Is the client a credit card holder?",
     ('Yes', 'No   '))
if cred_sel.rstrip()=='Yes':
    cred_sel=1
elif cred_sel.rstrip() == 'No':
    cred_sel=0









st.header('1- Exploratory data analysis')
st.write(" ")
st.write(" ")

outcome_perc=plot_df.groupby('Outcome').agg({'Outcome':'count'})
outcome_perc.rename(columns={'Outcome':'Ratio'},inplace=True)
tot_out=sum(outcome_perc['Ratio'])
outcome_perc['Ratio']=outcome_perc['Ratio']*100/tot_out

outcome_perc_pie = px.pie(outcome_perc, values='Ratio', names=['No Personal Loan','Personal Loan'], title='Campaign results')
outcome_perc_pie.update_traces(textfont_size=20,marker=dict(colors=['blue','red'],line=dict(color='#000000', width=2)))
#fig.show()


edlevel=plot_df.groupby(['Ed level','Outcome']).agg({'Ed level':'count'})
edlevel.reset_index(level='Outcome',inplace=True)
edlevel.rename(columns={'Ed level':'% of Total'},inplace=True)
sum_ed=edlevel['% of Total'].sum()
edlevel['% of Total']=edlevel['% of Total']*100/sum_ed
ed_lev_bar = px.bar(edlevel, x=edlevel.index, y='% of Total', title="Education level",color='Outcome',barmode="group")
ed_lev_bar.update_layout(xaxis = dict(
tickfont = dict(size=20)))
ed_lev_bar.update_traces( marker_line_color='rgb(8,48,107)',
                  marker_line_width=3)
#fig.show()
col1, col2 = st.columns(2)
with col1:
    st.subheader("A- Last year's conversion rate: 9.8% success rate ")
    st.write(outcome_perc_pie)
with col2:
    st.subheader("B- Education level distribution: the higher education, the better the conversion rate ")
    st.write(ed_lev_bar)   

df.Age.value_counts()
fig=plt.figure()
age_hist = px.histogram(plot_df, x="Age",nbins=5, title='Age breakdown',color='Outcome')

age_hist.update_layout(bargap=0.2,xaxis = dict(
tickfont = dict(size=13)))
age_hist.update_traces( marker_line_color='rgb(8,48,107)',
                  marker_line_width=3)




out_inc = px.box(plot_df,x='Outcome', y="Income",points="all",color='Outcome')
out_inc.update_traces( marker_line_color='rgb(8,48,107)',
                  marker_line_width=3)
out_inc.update_layout(xaxis = dict(tickfont = dict(size=20)))


col1, col2 = st.columns(2)
with col1:
    st.subheader("C- Age distribution: ")
    st.subheader('Majority of positive conversion between ages 30 and 60')
    st.write(age_hist,use_column_width=True)
with col2:
    st.subheader("D- Personal income:")
    st.subheader('Depositors with a higher income tend to have a higher income')
    st.write(out_inc,use_column_width=True)

fam=plot_df.groupby('Outcome').agg({'Family':np.mean})
fam.rename(columns={'Family':'Family size'},inplace =True)
fam_bar = px.bar(fam, x=fam.index, y='Family size', title="Family size",color=fam.index)
fam_bar.update_traces( marker_line_color='rgb(8,48,107)',
                  marker_line_width=3)
fam_bar.update_layout(xaxis = dict(tickfont = dict(size=20)))

col1,col2,col3=st.columns(3)

with col2:
    st.subheader("E-Family size: similar outcome among loan takers and non takers")
    st.write(fam_bar,use_column_width=True)
    
test={'Age':age_sel,'Experience':exp_sel,'Income':inc_sel,'ZIP Code':zip_sel,'Family':fam_sel,'CCAvg':cred_sel,'Education':edu_sel,'Mortgage':mor_sel,'Securities Account':sec_sel,'CD Account':cred_sel,'Online':on_sel,'CreditCard':cred_sel}
df=df.append(test,ignore_index=True)
df.loc[df['Age']<20,'age']='less than 20'
df.loc[(df['Age']>=20)&(df['Age']<30),'age']='between 20 and 30'
df.loc[(df['Age']>=30)&(df['Age']<40),'age']='between 30 and 40'
df.loc[(df['Age']>=40)&(df['Age']<50),'age']='between 40 and 50'
df.loc[(df['Age']>=50)&(df['Age']<60),'age']='between 50 and 60'
df.loc[(df['Age']>=50)&(df['Age']<60),'age']='between 50 and 60'
df.loc[(df['Age']>=60)&(df['Age']<70),'age']='between 60 and 70'
df.loc[(df['Age']>=70)&(df['Age']<80),'age']='between 70 and 80'


df.drop(columns=['Age'],axis=1,inplace=True)
st.subheader("E- Correlation between variables")
st.subheader(" We can see that our variables are not strongly correlated, thus we dont drop any")


heat_map = px.imshow(df[:-1].corr().round(3), text_auto=True, aspect="auto",width=1400, height=800)
heat_map.update_layout(xaxis = dict(tickfont = dict(size=18)))
heat_map.update_layout(yaxis = dict(tickfont = dict(size=18)))

heat_map.update_traces(textfont_size=18)
st.write(heat_map)









labelencoder = LabelEncoder()
df['age'] = labelencoder.fit_transform(df['age'])

#df.to_csv('Desktop/proj_7.csv')

y=df[['Personal Loan']][:-1]
x=df.copy()
x.drop(columns='Personal Loan',axis=1,inplace=True)
x_ml=x.iloc[-1,:]
x=x[:-1]


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)



####################################################################3



#"""  Gausian process classifier   """

from sklearn.gaussian_process import GaussianProcessClassifier
model = GaussianProcessClassifier()
model.fit(x_train, y_train)
y_pred= model.predict(x_test)
print("The accuracy of the Gausian process classifier is","{:.3f}".format(metrics.accuracy_score(y_pred,y_test)))
print("The precision of the Gausian process classifier is","{:.3f}".format(metrics.precision_score(y_test,y_pred)))
print("The recall score of the Gausian process classifier is","{:.3f}".format(metrics.recall_score(y_test,y_pred)))
print("The roc auc score of the Gausian process classifier is","{:.3f}".format(metrics.roc_auc_score(y_test,y_pred)))
print("The roc auc score of the Gausian process classifier  is","{:.3f}".format(metrics.roc_auc_score(y_test,y_pred)))
print(metrics.confusion_matrix(y_test, y_pred))
gpc_acc=metrics.accuracy_score(y_pred,y_test)
gpc_prec=metrics.precision_score(y_pred,y_test)
gpc_rec=metrics.recall_score(y_pred,y_test)
gpc_auc=metrics.roc_auc_score(y_pred,y_test)
gpc_conf=metrics.confusion_matrix(y_test, y_pred)
fpr, tpr, thresholds = metrics.roc_curve(y_test,y_pred)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr)
display.plot()
plt.title('GPC Roc Curve')
plt.show()



#""""   Quadratic discrimination analysis     """


clf = QuadraticDiscriminantAnalysis()
clf.fit(x_train, y_train)
y_pred=clf.predict(x_test)
print("The accuracy of the Quadratic discrimination analysis model is","{:.3f}".format(metrics.accuracy_score(y_pred,y_test)))
print("The precision of the Quadratic discrimination analysis model  is","{:.3f}".format(metrics.precision_score(y_test,y_pred)))
print("The recall score of the Quadratic discrimination analysis model  is","{:.3f}".format(metrics.recall_score(y_test,y_pred)))
print("The roc auc score of the Quadratic discrimination analysis model is","{:.3f}".format(metrics.roc_auc_score(y_test,y_pred)))
qda_acc=metrics.accuracy_score(y_pred,y_test)
qda_prec=metrics.precision_score(y_pred,y_test)
qda_rec=metrics.recall_score(y_pred,y_test)
qda_auc=metrics.roc_auc_score(y_pred,y_test)
qda_conf=metrics.confusion_matrix(y_test, y_pred)
print(metrics.confusion_matrix(y_test, y_pred))
fpr, tpr, thresholds = metrics.roc_curve(y_test,y_pred)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr)
display.plot()
plt.title("QDA Roc curve")
plt.show()


#"""   Decision tree    """



tree = DecisionTreeClassifier(max_depth = 8, random_state = 3)
tree.fit(x_train,y_train)
y_pred=tree.predict(x_test)
print("The accuracy of the Decision Tree is","{:.3f}".format(metrics.accuracy_score(y_pred,y_test)))
print("The precision of the Decision Tree is","{:.3f}".format(metrics.precision_score(y_test,y_pred)))
print("The recall score of the Decision Tree is","{:.3f}".format(metrics.recall_score(y_test,y_pred)))
print("The roc auc score of the Decision tree model is","{:.3f}".format(metrics.roc_auc_score(y_test,y_pred)))
dec_t_acc=metrics.accuracy_score(y_pred,y_test)
dec_t_prec=metrics.precision_score(y_pred,y_test)
dec_t_rec=metrics.recall_score(y_pred,y_test)
dec_t_auc=metrics.roc_auc_score(y_pred,y_test)
dec_t_conf=metrics.confusion_matrix(y_test, y_pred)
print(metrics.confusion_matrix(y_test, y_pred))
fpr, tpr, thresholds = metrics.roc_curve(y_test,y_pred)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr)
display.plot()
plt.title('Decision tree Roc curve')
plt.show()




#"""   Multinomial Naive Bayes """




x_train_mnb=x_train
y_train_mnb=y_train
x_test_mnb=x_test
y_test_mnb=y_test
enc = LabelEncoder()
x_train_mnb['Experience']=enc.fit_transform(x_train_mnb['Experience'])
x_test_mnb['Experience']=enc.fit_transform(x_test_mnb['Experience'])
classifier = MultinomialNB()
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print("The accuracy of the Multinomial Naive Bayes is","{:.3f}".format(metrics.accuracy_score(y_pred,y_test)))
print("The precision of the Multinomial Naive Bayes is","{:.3f}".format(metrics.precision_score(y_test,y_pred)))
print("The recall score of the Multinomial Naive Bayes is","{:.3f}".format(metrics.recall_score(y_test,y_pred)))
print("The roc auc score of the Multinomial Naive Bayes model is","{:.3f}".format(metrics.roc_auc_score(y_test,y_pred)))
mnb_acc=metrics.accuracy_score(y_pred,y_test)
mnb_prec=metrics.precision_score(y_pred,y_test)
mnb_rec=metrics.recall_score(y_pred,y_test)
mnb_auc=metrics.roc_auc_score(y_pred,y_test)
mnb_conf=metrics.confusion_matrix(y_test, y_pred)
print(metrics.confusion_matrix(y_test, y_pred))
fpr, tpr, thresholds = metrics.roc_curve(y_test,y_pred)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr)
display.plot()
plt.title('Multinomial NB Roc curve')
plt.show()


#""" Gradient Boosting Classifier """


clf= GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=1).fit(x_train, y_train)
y_pred = clf.predict(x_test)
print("The accuracy of the Gradient Boosting Classifier is","{:.3f}".format(metrics.accuracy_score(y_test,y_pred)))
print("The precision of the Gradient Boosting Classifier is","{:.3f}".format(metrics.precision_score(y_test,y_pred)))
print("The recall score of Gradient Boosting Classifier is","{:.3f}".format(metrics.recall_score(y_test,y_pred)))
print("The roc auc score of Gradient Boosting Classifier is","{:.3f}".format(metrics.roc_auc_score(y_test,y_pred)))
print("The roc auc score of Gradient Boosting Classifier  is","{:.3f}".format(metrics.roc_auc_score(y_test,y_pred)))
gbc_acc=metrics.accuracy_score(y_pred,y_test)
gbc_prec=metrics.precision_score(y_pred,y_test)
gbc_rec=metrics.recall_score(y_pred,y_test)
gbc_auc=metrics.roc_auc_score(y_pred,y_test)
gbc_conf=metrics.confusion_matrix(y_test, y_pred)
print(metrics.confusion_matrix(y_test, y_pred))
fpr, tpr, thresholds = metrics.roc_curve(y_test,y_pred)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr)
display.plot()
plt.title('GBC Roc curve')
plt.show()


#""" Bagging classifier with Gaussian process classifier  """


clf = BaggingClassifier(base_estimator=GaussianProcessClassifier(),n_estimators=10, random_state=1).fit(x_train, y_train)
y_pred=clf.predict(x_test)
print(metrics.accuracy_score(y_test,y_pred))
print(metrics.confusion_matrix(y_test, y_pred))
print("The accuracy of the Gausian process classifier is","{:.3f}".format(metrics.accuracy_score(y_pred,y_test)))
print("The precision of the Gausian process classifier is","{:.3f}".format(metrics.precision_score(y_test,y_pred)))
print("The recall score of the Gausian process classifier is","{:.3f}".format(metrics.recall_score(y_test,y_pred)))
print("The roc auc score of the Gausian process classifier is","{:.3f}".format(metrics.roc_auc_score(y_test,y_pred)))
print("The roc auc score of the Gausian process classifier  is","{:.3f}".format(metrics.roc_auc_score(y_test,y_pred)))
print(metrics.confusion_matrix(y_test, y_pred))
bg_gpc_acc=metrics.accuracy_score(y_pred,y_test)
bg_gpc_prec=metrics.precision_score(y_pred,y_test)
bg_gpc_rec=metrics.recall_score(y_pred,y_test)
bg_gpc_auc=metrics.roc_auc_score(y_pred,y_test)
bg_gpc_conf=metrics.confusion_matrix(y_test, y_pred)
fpr, tpr, thresholds = metrics.roc_curve(y_test,y_pred)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr)
display.plot()
plt.title('GPC +  Bagging classifier Roc curve')
plt.show()



#"""   Bagging Classifier with QDA  """


clf = BaggingClassifier(base_estimator=QuadraticDiscriminantAnalysis(),n_estimators=10, random_state=1).fit(x_train, y_train)
y_pred=clf.predict(x_test)
print("The accuracy of the Quadratic discrimination analysis model is","{:.3f}".format(metrics.accuracy_score(y_pred,y_test)))
print("The precision of the Quadratic discrimination analysis model  is","{:.3f}".format(metrics.precision_score(y_test,y_pred)))
print("The recall score of the Quadratic discrimination analysis model  is","{:.3f}".format(metrics.recall_score(y_test,y_pred)))
print("The roc auc score of the Quadratic discrimination analysis model is","{:.3f}".format(metrics.roc_auc_score(y_test,y_pred)))
bg_qda_acc=metrics.accuracy_score(y_pred,y_test)
bg_qda_prec=metrics.precision_score(y_pred,y_test)
bg_qda_rec=metrics.recall_score(y_pred,y_test)
bg_qda_auc=metrics.roc_auc_score(y_pred,y_test)
bg_qda_conf=metrics.confusion_matrix(y_test, y_pred)
print(metrics.confusion_matrix(y_test, y_pred))
fpr, tpr, thresholds = metrics.roc_curve(y_test,y_pred)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr)
display.plot()
plt.title('QDA +  Bagging classifier Roc curve')
plt.show()



#"""   Bagging Classifier with Decision tree  """


from sklearn.tree import DecisionTreeClassifier
clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=10,random_state=1).fit(x_train, y_train)
y_pred=clf.predict(x_test)
print("The accuracy of the Decision Tree is","{:.3f}".format(metrics.accuracy_score(y_pred,y_test)))
print("The precision of the Decision Tree is","{:.3f}".format(metrics.precision_score(y_test,y_pred)))
print("The recall score of the Decision Tree is","{:.3f}".format(metrics.recall_score(y_test,y_pred)))
print("The roc auc score of the Decision tree model is","{:.3f}".format(metrics.roc_auc_score(y_test,y_pred)))
bg_dec_t_acc=metrics.accuracy_score(y_pred,y_test)
bg_dec_t_prec=metrics.precision_score(y_pred,y_test)
bg_dec_t_rec=metrics.recall_score(y_pred,y_test)
bg_dec_t_auc=metrics.roc_auc_score(y_pred,y_test)
bg_dec_t_conf=metrics.confusion_matrix(y_test, y_pred)
print(metrics.confusion_matrix(y_test, y_pred))
fpr, tpr, thresholds = metrics.roc_curve(y_test,y_pred)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr)
display.plot()
plt.title('Decision tree +  Bagging classifier Roc curve')
plt.show()


#"""   Bagging Classifier with Multinomial Naive Bayes """

x_train_mnb=x_train
y_train_mnb=y_train
x_test_mnb=x_test
y_test_mnb=y_test
enc = LabelEncoder()
x_train_mnb['Experience']=enc.fit_transform(x_train_mnb['Experience'])
x_test_mnb['Experience']=enc.fit_transform(x_test_mnb['Experience'])
clf = BaggingClassifier(base_estimator=MultinomialNB(),n_estimators=10, random_state=1).fit(x_train_mnb, y_train_mnb)
y_pred=clf.predict(x_test_mnb)
print(metrics.accuracy_score(y_test_mnb,y_pred))
print(metrics.confusion_matrix(y_test_mnb, y_pred))
print("The accuracy of the MultinomialNB classifier is","{:.3f}".format(metrics.accuracy_score(y_pred,y_test)))
print("The precision of the MultinomialNB classifier is","{:.3f}".format(metrics.precision_score(y_test,y_pred)))
print("The recall score of the MultinomialNB classifier is","{:.3f}".format(metrics.recall_score(y_test,y_pred)))
print("The roc auc score of the MultinomialNB classifier is","{:.3f}".format(metrics.roc_auc_score(y_test,y_pred)))
print("The roc auc score of the MultinomialNB classifier  is","{:.3f}".format(metrics.roc_auc_score(y_test,y_pred)))
print(metrics.confusion_matrix(y_test, y_pred))
bg_mnb_acc=metrics.accuracy_score(y_pred,y_test)
bg_mnb_prec=metrics.precision_score(y_pred,y_test)
bg_mnb_rec=metrics.recall_score(y_pred,y_test)
bg_mnb_auc=metrics.roc_auc_score(y_pred,y_test)
bg_mnb_conf=metrics.confusion_matrix(y_test, y_pred)
fpr, tpr, thresholds = metrics.roc_curve(y_test,y_pred)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr)
display.plot()
plt.title('MNB +  Bagging classifier Roc curve')
plt.show()



#""" Bagging classifier with Gradient Boosting Classifier """


bgc_gbc = BaggingClassifier(base_estimator=GradientBoostingClassifier(),n_estimators=10, random_state=1).fit(x_train, y_train)
y_pred=bgc_gbc.predict(x_test)
print(metrics.accuracy_score(y_test,y_pred))
print(metrics.confusion_matrix(y_test, y_pred))
print("The accuracy of the Gradient Boosting Classifier classifier is","{:.3f}".format(metrics.accuracy_score(y_pred,y_test)))
print("The precision of the Gradient Boosting Classifier classifier is","{:.3f}".format(metrics.precision_score(y_test,y_pred)))
print("The recall score of the Gradient Boosting Classifier classifier is","{:.3f}".format(metrics.recall_score(y_test,y_pred)))
print("The roc auc score of the Gradient Boosting Classifier classifier is","{:.3f}".format(metrics.roc_auc_score(y_test,y_pred)))
print("The roc auc score of the Gradient Boosting Classifier classifier  is","{:.3f}".format(metrics.roc_auc_score(y_test,y_pred)))
print(metrics.confusion_matrix(y_test, y_pred))
bg_gbc_acc=metrics.accuracy_score(y_pred,y_test)
bg_gbc_prec=metrics.precision_score(y_pred,y_test)
bg_gbc_rec=metrics.recall_score(y_pred,y_test)
bg_gbc_auc=metrics.roc_auc_score(y_pred,y_test)
bg_gbc_conf=metrics.confusion_matrix(y_test, y_pred)
fpr, tpr, thresholds = metrics.roc_curve(y_test,y_pred)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr)
display.plot()
plt.title('GBC +  Bagging classifier Roc curve')
x_ml=np.array(x_ml).reshape(1,-1)
result=bgc_gbc.predict(x_ml)


########
st.header("2- Model training")
st.subheader('We will be testing 5 models: ')
st.write('1- Gaussian Process classifier : a non-parametric classification method based on Bayesian methodology, final classification determined by best fit graphs that gurantee smoothness. Makes predictions with uncertainties.')
st.write('2- Quadratic Discriminant analysis: A statsistical classifier that uses quadratic decision surfaces to seperate classes')
st.write('3- Decision Tree classifier: From its name, creates rules that branch until a final classification is made')
st.write('4- Multinomial Naive Bayes: a collection of robabilistic learning methods that assume features are not related')
st.write('5-Gradient Boosting classifier: A set of machine learning algorithms that use several weaker models and combine them into a strong big one with highly predictive output')
st.write('We will also use a bagging classifier. A bagging classifiier is applied to each model and randomizes training data sets to reduce overfitting')
st.write(" ")
st.subheader('To compare between models, we compare their: accuracy, precision, recall and auc score (measures degree of seperation 0.5 (random) to 1 (perfect)')




comp=pd.DataFrame({'Classifier':['Gaussian Process Classifier','Quadratic Discriminant Analysis','Decision Tree Classifier','Multinomial NB','Gradient Boosting Classifier'],'Accuracy':[gpc_acc,qda_acc,dec_t_acc,mnb_acc,gbc_acc],'Precision':[gpc_prec,qda_prec,dec_t_prec,mnb_prec,gbc_prec],'Recall':[gpc_rec,qda_rec,dec_t_rec,mnb_rec,gbc_rec],'AUC score':[gpc_auc,qda_auc,dec_t_auc,mnb_auc,gbc_auc]})
comp.set_index('Classifier',inplace=True)

comp_bg=pd.DataFrame({'Classifier':['Gaussian Process Classifier','Quadratic Discriminant Analysis','Decision Tree Classifier','Multinomial NB','Gradient Boosting Classifier'],'Accuracy':[bg_gpc_acc,bg_qda_acc,bg_dec_t_acc,bg_mnb_acc,bg_gbc_acc],'Precision':[bg_gpc_prec,bg_qda_prec,bg_dec_t_prec,bg_mnb_prec,bg_gbc_prec],'Recall':[bg_gpc_rec,bg_qda_rec,bg_dec_t_rec,bg_mnb_rec,bg_gbc_rec],'AUC score':[bg_gpc_auc,bg_qda_auc,bg_dec_t_auc,bg_mnb_auc,bg_gbc_auc]})
comp_bg.set_index('Classifier',inplace=True)

comp_bar = px.bar(comp, title="Classifier comparisons",barmode="group",width=800, height=800)
comp_bar.update_traces( marker_line_color='rgb(8,48,107)',
                  marker_line_width=3)
comp_bar.update_layout(xaxis = dict(tickfont = dict(size=14)))
comp_bar.update_layout(yaxis = dict(tickfont = dict(size=14)))
col1, col2= st.columns(2)
with col1:
    st.write(comp_bar)


compbg_bar = px.bar(comp_bg, title="Bagged classifier comparisons",barmode="group",width=800, height=800)
compbg_bar.update_traces( marker_line_color='rgb(8,48,107)',
                  marker_line_width=3)
compbg_bar.update_layout(xaxis = dict(tickfont = dict(size=14)))
compbg_bar.update_layout(yaxis = dict(tickfont = dict(size=14)))
with col2:
    st.write(compbg_bar)
st.write(' ')
st.subheader('The Gradient boosting classifier seems to perform the best')
conf_mat = px.imshow(bg_gbc_conf, text_auto=True, aspect="auto",width=1400, height=800)
conf_mat.update_layout(xaxis = dict(tickfont = dict(size=18)))
conf_mat.update_layout(yaxis = dict(tickfont = dict(size=18)))

conf_mat.update_traces(textfont_size=30)
st.write(conf_mat)

st.subheader('For our model predictions, we will rely on the gradient boosting classifier')
st.subheader('On the left, you can input the characteristics of the client and the model will predict whether the client will take a personal loan or not !')

if result[0]==0:
    show='Will not take a Personal loan'
elif result[0]==1:
    show='Will take a Personal loan'
st.sidebar.header('👉 '+ show)
    
