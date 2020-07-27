
# # Lead Scoring 

# ## Logistic Regression

# ## Problem Statement
# An education company named __X Education__ sells online courses to industry professionals. On any given day, many professionals who are interested in the courses land on their website and browse for courses. 
# 
#  
# 
# The company markets its courses on several websites and search engines like Google. Once these people land on the website, they might browse the courses or fill up a form for the course or watch some videos. <br>
# 
# __When these people fill up a form providing their email address or phone number, they are classified to be a lead. Moreover, the company also gets leads through past referrals.__<br>
# 
# Once these leads are acquired, employees from the sales team start making calls, writing emails, etc. Through this process, some of the leads get converted while most do not. __The typical lead conversion rate at X education is around 30%.__
#  
# 
# Now, although X Education gets a lot of leads, its lead conversion rate is very poor. For example, if, say, they acquire 100 leads in a day, only about 30 of them are converted. To make this process more efficient, the company wishes to identify the most potential leads, also known as __‘Hot Leads’__. <br>
# 
# If they successfully identify this set of leads, the lead conversion rate should go up as the sales team will now be focusing more on communicating with the potential leads rather than making calls to everyone. A typical lead conversion process can be represented using the following funnel:
# ![image.jpg](attachment:image.jpg)
# 
# 
# 
# __Lead Conversion Process__ - Demonstrated as a funnel
# As you can see, there are a lot of leads generated in the initial stage (top) but only a few of them come out as paying customers from the bottom.<br>
# 
# In the middle stage, you need to nurture the potential leads well (i.e. educating the leads about the product, constantly communicating etc. ) in order to get a higher lead conversion. 
# 
# X Education has appointed you to help them select the most promising leads, i.e. the leads that are most likely to convert into paying customers. <br>
# The company requires you to build a model wherein you need to assign a lead score to each of the leads such that the customers with higher lead score have a higher conversion chance and the customers with lower lead score have a lower conversion chance.
# 
# __The CEO, in particular, has given a ballpark of the target lead conversion rate to be around 80%.__
# 
# 
#  
# 
# ### Data
# 
# You have been provided with a leads dataset from the past with around 9000 data points. This dataset consists of various attributes such as Lead Source, Total Time Spent on Website, Total Visits, Last Activity, etc. which may or may not be useful in ultimately deciding whether a lead will be converted or not. The target variable, in this case, is the column ‘Converted’ which tells whether a past lead was converted or not wherein 1 means it was converted and 0 means it wasn’t converted.
# 
# Another thing that you also need to check out for are the levels present in the categorical variables.<br>
# 
# __Many of the categorical variables have a level called 'Select' which needs to be handled because it is as good as a null value.__
# 
#  
# 
# ### Goal
# 
# 
# 1. Build a logistic regression model to assign a lead score between 0 and 100 to each of the leads which can be used by the company to target potential leads. A higher score would mean that the lead is hot, i.e. is most likely to convert whereas a lower score would mean that the lead is cold and will mostly not get converted.

# Supress Warnings
import warnings
warnings.filterwarnings('ignore')

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# visulaisation
from matplotlib.pyplot import xticks
get_ipython().run_line_magic('matplotlib', 'inline')

# Data display coustomization
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)


# ## Data Preparation

# ### Data Loading


data = pd.DataFrame(pd.read_csv('Leads.csv'))
data.head(5) 

#checking duplicates
sum(data.duplicated(subset = 'Prospect ID')) == 0
# No duplicate values


# ### Data Inspection

data.shape

data.info()

data.describe()


# ### Data Cleaning

# As we can observe that there are select values for many column.
#This is because customer did not select any option from the list, hence it shows select.
# Select values are as good as NULL.

# Converting 'Select' values to NaN.
data = data.replace('Select', np.nan)

data.isnull().sum()

round(100*(data.isnull().sum()/len(data.index)), 2)

# we will drop the columns having more than 70% NA values.
data = data.drop(data.loc[:,list(round(100*(data.isnull().sum()/len(data.index)), 2)>70)].columns, 1)

# Taking care of the Lead Quality column
# Lead Quality: Indicates the quality of lead based on the data and intuition the the employee who has been assigned to the lead

data['Lead Quality'].describe()

sns.countplot(data['Lead Quality'])

# As Lead quality is based on the intution of employee, so if left blank we can impute 'Not Sure' in NaN safely.
data['Lead Quality'] = data['Lead Quality'].replace(np.nan, 'Not Sure')

sns.countplot(data['Lead Quality'])

# Asymmetrique Activity Index  |
# Asymmetrique Profile Index   \   An index and score assigned to each customer
# Asymmetrique Activity Score  |    based on their activity and their profile
# Asymmetrique Profile Score   \

fig, axs = plt.subplots(2,2, figsize = (10,7.5))
plt1 = sns.countplot(data['Asymmetrique Activity Index'], ax = axs[0,0])
plt2 = sns.boxplot(data['Asymmetrique Activity Score'], ax = axs[0,1])
plt3 = sns.countplot(data['Asymmetrique Profile Index'], ax = axs[1,0])
plt4 = sns.boxplot(data['Asymmetrique Profile Score'], ax = axs[1,1])
plt.tight_layout()

# There is too much variation in thes parameters so its not reliable to impute any value in it. 
# 45% null values means we need to drop these columns.

data = data.drop(['Asymmetrique Activity Index','Asymmetrique Activity Score','Asymmetrique Profile Index','Asymmetrique Profile Score'],1)

round(100*(data.isnull().sum()/len(data.index)), 2)

#  Taking care of the City column

data.City.describe()

sns.countplot(data.City)
xticks(rotation = 90)

# Around 60% of the data is Mumbai so we can impute Mumbai in the missing values.
data['City'] = data['City'].replace(np.nan, 'Mumbai')

#  Taking care of Specailization column 

data.Specialization.describe()

sns.countplot(data.Specialization)
xticks(rotation = 90)

# It maybe the case that lead has not entered any specialization if his/her option is not availabe on the list,
#  may not have any specialization or is a student.
# Hence we can make a category "Others" for missing values. 

data['Specialization'] = data['Specialization'].replace(np.nan, 'Others')

round(100*(data.isnull().sum()/len(data.index)), 2)



# Taking care of Tags column

data.Tags.describe()

fig, axs = plt.subplots(figsize = (15,7.5))
sns.countplot(data.Tags)
xticks(rotation = 90)

# Blanks in the tag column may be imputed by 'Will revert after reading the email'.

data['Tags'] = data['Tags'].replace(np.nan, 'Will revert after reading the email')



# Taking care of  What matters most to you in choosing a course column

data['What matters most to you in choosing a course'].describe()

# Blanks in the this column may be imputed by 'Better Career Prospects'.

data['What matters most to you in choosing a course'] = data['What matters most to you in choosing a course'].replace(np.nan, 'Better Career Prospects')



# Taking care of Occupation column

data['What is your current occupation'].describe()

# 86% entries are of Unemployed so we can impute "Unemployed" in it.

data['What is your current occupation'] = data['What is your current occupation'].replace(np.nan, 'Unemployed')



#  Taking care of Country column

# Country is India for most values so let's impute the same in missing values.
data['Country'] = data['Country'].replace(np.nan, 'India')

round(100*(data.isnull().sum()/len(data.index)), 2)



# Rest missing values are under 2% so we can drop these rows.
data.dropna(inplace = True)



round(100*(data.isnull().sum()/len(data.index)), 2)

data.to_csv('Leads_cleaned')


# #### Now Data is clean and we can start with the analysis part

# # Exploratory Data Analytics

# ## Univariate Analysis

# ### Converted

# Converted is the target variable, Indicates whether a lead has been successfully converted (1) or not (0).



Converted = (sum(data['Converted'])/len(data['Converted'].index))*100
Converted


# ### Lead Origin

sns.countplot(x = "Lead Origin", hue = "Converted", data = data)
xticks(rotation = 90)


# #### Inference
# 1. API and Landing Page Submission have 30-35% conversion rate but count of lead originated from them are considerable.
# 2. Lead Add Form has more than 90% conversion rate but count of lead are not very high.
# 3. Lead Import are very less in count.<br>
# 
# 
# __To improve overall lead conversion rate, we need to focus more on improving lead converion of API and Landing Page Submission origin and generate more leads from Lead Add Form.__

# ### Lead Source

fig, axs = plt.subplots(figsize = (15,7.5))
sns.countplot(x = "Lead Source", hue = "Converted", data = data)
xticks(rotation = 90)


data['Lead Source'] = data['Lead Source'].replace(['google'], 'Google')
data['Lead Source'] = data['Lead Source'].replace(['Click2call', 'Live Chat', 'NC_EDM', 'Pay per Click Ads', 'Press_Release',
  'Social Media', 'WeLearn', 'bing', 'blog', 'testone', 'welearnblog_Home', 'youtubechannel'], 'Others')


sns.countplot(x = "Lead Source", hue = "Converted", data = data)
xticks(rotation = 90)


# #### Inference
# 1. Google and Direct traffic generates maximum number of leads.
# 2. Conversion Rate of reference leads and leads through welingak website is high.
# 
# __To improve overall lead conversion rate, focus should be on improving lead converion of olark chat, organic search, direct traffic, and google  leads and generate more leads from reference and welingak website.__




# ### Do Not Email & Do Not Call
fig, axs = plt.subplots(1,2,figsize = (15,7.5))
sns.countplot(x = "Do Not Email", hue = "Converted", data = data, ax = axs[0])
sns.countplot(x = "Do Not Call", hue = "Converted", data = data, ax = axs[1])


# ### Total Visits

data['TotalVisits'].describe(percentiles=[0.05,.25, .5, .75, .90, .95, .99])

sns.boxplot(data['TotalVisits'])

# As we can see there are a number of outliers in the data.
# We will cap the outliers to 95% value for analysis.

percentiles = data['TotalVisits'].quantile([0.05,0.95]).values
data['TotalVisits'][data['TotalVisits'] <= percentiles[0]] = percentiles[0]
data['TotalVisits'][data['TotalVisits'] >= percentiles[1]] = percentiles[1]

sns.boxplot(data['TotalVisits'])

sns.boxplot(y = 'TotalVisits', x = 'Converted', data = data)


# #### Inference
# 1. Median for converted and not converted leads are the same.
# 
# __Nothng conclusive can be said on the basis of Total Visits.__

# ### Total time spent on website


data['Total Time Spent on Website'].describe()


sns.boxplot(data['Total Time Spent on Website'])

sns.boxplot(y = 'Total Time Spent on Website', x = 'Converted', data = data)


# #### Inference
# 1. Leads spending more time on the weblise are more likely to be converted.
# 
# __Website should be made more engaging to make leads spend more time.__



# ### Page views per visit

data['Page Views Per Visit'].describe()

sns.boxplot(data['Page Views Per Visit'])

# As we can see there are a number of outliers in the data.
# We will cap the outliers to 95% value for analysis.

percentiles = data['Page Views Per Visit'].quantile([0.05,0.95]).values
data['Page Views Per Visit'][data['Page Views Per Visit'] <= percentiles[0]] = percentiles[0]
data['Page Views Per Visit'][data['Page Views Per Visit'] >= percentiles[1]] = percentiles[1]

sns.boxplot(data['Page Views Per Visit'])

sns.boxplot(y = 'Page Views Per Visit', x = 'Converted', data = data)

# #### Inference
# 1. Median for converted and unconverted leads is the same.
# 
# __Nothing can be said specifically for lead conversion from Page Views Per Visit __




# ### Last Activity

data['Last Activity'].describe()

fig, axs = plt.subplots(figsize = (15,5))
sns.countplot(data['Last Activity'])
xticks(rotation = 90)

# Let's keep considerable last activities as such and club all others to "Other_Activity"
data['Last Activity'] = data['Last Activity'].replace(['Had a Phone Conversation', 'View in browser link Clicked', 
                                                       'Visited Booth in Tradeshow', 'Approached upfront',
                                                       'Resubscribed to emails','Email Received', 'Email Marked Spam'], 'Other_Activity')


fig, axs = plt.subplots(figsize = (10,5))
sns.countplot(x = "Last Activity", hue = "Converted", data = data)
xticks(rotation = 90)


# #### Inference
# 1. Most of the lead have their Email opened as their last activity.
# 2. Conversion rate for leads with last activity as SMS Sent is almost 60%.b




# ### Country

data.Country.describe()

# ### Inference
# Most values are 'India' no such inference can be drawn



# ### Specialization

data.Specialization.describe()
fig, axs = plt.subplots(figsize = (15,5))
sns.countplot(data.Specialization)
xticks(rotation=90)

data['Specialization'] = data['Specialization'].replace(['Others'], 'Other_Specialization')

fig, axs = plt.subplots(figsize = (15,5))
sns.countplot(x = "Specialization", hue = "Converted", data = data)
xticks(rotation = 90)

# #### Inference
# 1. Focus should be more on the Specialization with high conversion rate.



# ### Occupation

data['What is your current occupation'].describe()


fig, axs = plt.subplots(figsize = (10,5))
sns.countplot(data['What is your current occupation'])
xticks(rotation = 90)

data['What is your current occupation'] = data['What is your current occupation'].replace(['Other'], 'Other_Occupation')

fig, axs = plt.subplots(figsize = (10,5))
sns.countplot(x = "What is your current occupation", hue = "Converted", data = data)
xticks(rotation = 90)


#  ### Inference
# 1. Working Professionals going for the course have high chances of joining it.
# 2. Unemployed leads are the most in numbers but has around 30-35% conversion rate.



# ### What matters most to you in choosing a course

data['What matters most to you in choosing a course'].describe()

# ### Inference
# Most entries are 'Better Career Prospects'.
# No Inference can be drawn with this parameter.



# ### Search

data.Search.describe()


# ### Inference
# Most entries are 'No'.
# No Inference can be drawn with this parameter.



# ### Magazine

data.Magazine.describe()

# ### Inference
# Most entries are 'No'.
# No Inference can be drawn with this parameter.



# ### Newspaper Article

data['Newspaper Article'].describe()

# ### Inference
# Most entries are 'No'.
# No Inference can be drawn with this parameter.



# ### X Education Forums

data['X Education Forums'].describe()

# ### Inference
# Most entries are 'No'.
# No Inference can be drawn with this parameter.



# ### Newspaper

data['Newspaper'].describe()

# ### Inference
# Most entries are 'No'.
# No Inference can be drawn with this parameter.



# ### Digital Advertisement

data['Digital Advertisement'].describe()

# ### Inference
# Most entries are 'No'.
# No Inference can be drawn with this parameter.



# ### Through Recommendations

data['Through Recommendations'].describe()

# ### Inference
# Most entries are 'No'.
# No Inference can be drawn with this parameter.



# ### Receive More Updates About Our Courses

data['Receive More Updates About Our Courses'].describe()

# ### Inference
# Most entries are 'No'.
# No Inference can be drawn with this parameter.



# #### Tags

data.Tags.describe()

fig, axs = plt.subplots(figsize = (15,5))
sns.countplot(data['Tags'])
xticks(rotation = 90)

# Let's keep considerable last activities as such and club all others to "Other_Activity"
data['Tags'] = data['Tags'].replace(['In confusion whether part time or DLP', 'in touch with EINS','Diploma holder (Not Eligible)',
                                     'Approached upfront','Graduation in progress','number not provided', 'opp hangup','Still Thinking',
                                    'Lost to Others','Shall take in the next coming month','Lateral student','Interested in Next batch',
                                    'Recognition issue (DEC approval)','Want to take admission but has financial problems',
                                    'University not recognized'], 'Other_Tags')


fig, axs = plt.subplots(figsize = (10,5))
sns.countplot(x = "Tags", hue = "Converted", data = data)
xticks(rotation = 90)

# ### Inference
# No Interference



# ### Lead Quality

data['Lead Quality'].describe()

fig, axs = plt.subplots(figsize = (10,5))
sns.countplot(x = "Lead Quality", hue = "Converted", data = data)
xticks(rotation = 90)

# ### Interference 
# No INterference



# ### Update me on Supply Chain Content

data['Update me on Supply Chain Content'].describe()

# ### Inference
# Most entries are 'No'.
# No Inference can be drawn with this parameter.



# ### Get updates on DM Content

data['Get updates on DM Content'].describe()

# ### Inference
# Most entries are 'No'.
# No Inference can be drawn with this parameter.



# ### I agree to pay the amount through cheque

data['I agree to pay the amount through cheque'].describe()

# ### Inference
# Most entries are 'No'.
# No Inference can be drawn with this parameter.



# ### A free copy of Mastering The Interview

data['A free copy of Mastering The Interview'].describe()

# ### Inference
# Most entries are 'No'.
# No Inference can be drawn with this parameter.



# ### City

data.City.describe()

fig, axs = plt.subplots(figsize = (10,5))
sns.countplot(x = "City", hue = "Converted", data = data)
xticks(rotation = 90)

# ### Inference
# Most leads are from mumbai with around 30% conversion rate.



# ### Last Notable Activity

data['Last Notable Activity'].describe()

fig, axs = plt.subplots(figsize = (10,5))
sns.countplot(x = "Last Notable Activity", hue = "Converted", data = data)
xticks(rotation = 90)

# ## Results
# 
# __Based on the univariate analysis we have seen that many columns are not adding any information to the model, heance we can drop them for frther analysis__

data = data.drop(['Lead Number','What matters most to you in choosing a course','Search','Magazine','Newspaper Article','X Education Forums','Newspaper',
           'Digital Advertisement','Through Recommendations','Receive More Updates About Our Courses','Update me on Supply Chain Content',
           'Get updates on DM Content','I agree to pay the amount through cheque','A free copy of Mastering The Interview','Country'],1)


data.shape

data.head()


# ### Data Preparation

# #### Converting some binary variables (Yes/No) to 1/0

# In[111]:


# List of variables to map

varlist =  ['Do Not Email', 'Do Not Call']

# Defining the map function
def binary_map(x):
    return x.map({'Yes': 1, "No": 0})

# Applying the function to the housing list
data[varlist] = data[varlist].apply(binary_map)


# #### For categorical variables with multiple levels, create dummy features (one-hot encoded)

# In[112]:


# Creating a dummy variable for some of the categorical variables and dropping the first one.
dummy1 = pd.get_dummies(data[['Lead Origin', 'Lead Source', 'Last Activity', 'Specialization','What is your current occupation',
                              'Tags','Lead Quality','City','Last Notable Activity']], drop_first=True)
dummy1.head()


# In[113]:


# Adding the results to the master dataframe
data = pd.concat([data, dummy1], axis=1)
data.head()


# In[114]:


data = data.drop(['Lead Origin', 'Lead Source', 'Last Activity', 'Specialization','What is your current occupation','Tags','Lead Quality','City','Last Notable Activity'], axis = 1)


# In[115]:


data.head()


# In[116]:


from sklearn.model_selection import train_test_split

# Putting feature variable to X
X = data.drop(['Prospect ID','Converted'], axis=1)


# In[117]:


X.head()


# In[118]:


# Putting response variable to y
y = data['Converted']

y.head()


# In[119]:


# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# ### Step 5: Feature Scaling

# In[120]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.fit_transform(X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])

X_train.head()


# In[121]:


# Checking the Churn Rate
Converted = (sum(data['Converted'])/len(data['Converted'].index))*100
Converted


# We have almost 38% conversion

# ### Feature Selection Using RFE

# In[124]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

from sklearn.feature_selection import RFE
rfe = RFE(logreg, 15)             # running RFE with 15 variables as output
rfe = rfe.fit(X_train, y_train)


# In[125]:


rfe.support_


# In[126]:


list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[127]:


col = X_train.columns[rfe.support_]
col


# In[128]:


X_train.columns[~rfe.support_]


# In[130]:


X_train_sm = X_train[col]
X_train_sm


# In[131]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(penalty='l2', max_iter=200, solver='lbfgs',random_state = 0 )
classifier.fit(X_train_sm,y_train)


# In[134]:


y_train_pred=classifier.predict(X_train_sm)


# ##### Creating a dataframe with the actual churn flag and the predicted probabilities

# In[135]:


y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_prob':y_train_pred})
y_train_pred_final['Prospect ID'] = y_train.index
y_train_pred_final.head()


# ##### Creating new column 'predicted' with 1 if Churn_Prob > 0.5 else 0

# In[137]:


y_train_pred_final['predicted'] = y_train_pred_final.Converted_prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()


# In[138]:


from sklearn import metrics

# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted )
print(confusion)


# In[ ]:


# Predicted     not_churn    churn
# Actual
# not_churn        3270      365
# churn            579       708  


# In[139]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted))


# ## Metrics beyond simply accuracy

# In[140]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[141]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[142]:


# Let us calculate specificity
TN / float(TN+FP)


# In[143]:


# Calculate false postive rate - predicting churn when customer does not have churned
print(FP/ float(TN+FP))


# In[144]:


# positive predictive value 
print (TP / float(TP+FP))


# In[ ]:


# Negative predictive value
print (TN / float(TN+ FN))


# ### Step 9: Plotting the ROC Curve

# An ROC curve demonstrates several things:
# 
# - It shows the tradeoff between sensitivity and specificity (any increase in sensitivity will be accompanied by a decrease in specificity).
# - The closer the curve follows the left-hand border and then the top border of the ROC space, the more accurate the test.
# - The closer the curve comes to the 45-degree diagonal of the ROC space, the less accurate the test.

# In[145]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[146]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Converted_prob, drop_intermediate = False )


# In[147]:


draw_roc(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)


# In[148]:


X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.fit_transform(X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])

X_train.head()


# In[149]:


X_test = X_test[col]
X_test.head()


# In[151]:


y_test_pred = classifier.predict(X_test)


# In[152]:


y_test_pred[:10]


# In[153]:


print(metrics.accuracy_score(y_test, y_test_pred))


# In[154]:


from sklearn.neighbors import KNeighborsClassifier
classifier1 = KNeighborsClassifier(n_neighbors=5,weights='uniform',metric='minkowski')
classifier1.fit(X_train_sm,y_train)


# In[158]:


y_test_pred1 = classifier1.predict(X_test)


# In[159]:


print(metrics.accuracy_score(y_test, y_test_pred1))


# In[163]:


from sklearn.svm import SVC
classifier2 = SVC(kernel = 'rbf',gamma='scale',C = 1.0, random_state = 0)
classifier2.fit(X_train_sm, y_train)


# In[164]:


y_test_pred2 = classifier2.predict(X_test)


# In[165]:


print(metrics.accuracy_score(y_test, y_test_pred2))


# In[166]:


from sklearn.ensemble import RandomForestClassifier
classifier3 = RandomForestClassifier(n_estimators=150)
classifier3.fit(X_train_sm, y_train)


# In[167]:


y_test_pred3 = classifier3.predict(X_test)


# In[168]:


print(metrics.accuracy_score(y_test, y_test_pred3))


# In[ ]:




