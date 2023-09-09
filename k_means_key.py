'''
# K-Means Clustering Algorithm - Data Mining (Machine Learning) Unsupervised learning Algorithm

# Business Problem Statement: The insurance company wants to analyze their customer’s behavior to strategies offers to increase customer loyalty.

# `CRISP-ML(Q)` process model describes six phases:
# 1. Business and Data Understanding
# 2. Data Preparation
# 3. Model Building
# 4. Evaluation
# 5. Deployment
# 6. Monitoring and Maintenance
'''
# Objective(s): Maximize the customer retention rate 
# Constraints: Minimize the 

'''Success Criteria'''

# Business Success Criteria: Reduce the churnrate from anywhere between 10% to 20%
# ML Success Criteria: Achieve Silhoutte coefficient of atleast 0.5
# Economic Success Criteria: Insurance company  will see an increase in revenues by atleast 30%

# **Proposed Plan:**
# Grouping the available customers  will allow to understand the characteristics of each group.


'''
# ## Data Collection
# Data: Data of 9000 customers belongs to usa has been collected

## Data Dictionary:
    # - Dataset contains 9000 customer details
    # - 24  features are recorded for each customer
# Description:
    Customer: Unique identifier for each customer.
    State: The state where the customer lives.
    Customer Lifetime Value: The total amount of money the customer is expected to spend on the company's products/services during their lifetime.
    Response: Whether the customer responded to the company's marketing efforts.
    Coverage: The type of insurance coverage the customer has.
    Education: The customer's highest level of education.
    Effective To Date: The date when the insurance policy became effective.
    Employment Status: The customer's current employment status.
    Gender: The customer's gender.
    Income: The customer's annual income.
    Location Code: The zip code where the customer lives.
    Marital Status: The customer's marital status.
    Monthly Premium Auto: The amount the customer pays each month for their insurance premium.
    Months Since Last Claim: The number of months since the customer last made an insurance claim.
    Months Since Policy Inception: The number of months since the customer first purchased an insurance policy.
    Number of Open Complaints: The number of open complaints the customer has with the company.
    Number of Policies: The number of insurance policies the customer has with the company.
    Policy Type: The type of insurance policy the customer has.
    Policy: The unique identifier for the customer's insurance policy.
    Renew Offer Type: The type of offer the company made to the customer to renew their policy.
    Sales Channel: The channel through which the customer purchased their insurance policy.
    Total Claim Amount: The total amount of money the customer has claimed from the company for insurance purposes.
    Vehicle Class: The type of vehicle the customer has insured.
    Vehicle Size: The size of the customer's vehicle.
'''
# Importing required packages

import pandas as pd 
import sweetviz
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from feature_engine.outliers import Winsorizer
from sklearn.cluster import KMeans,DBSCAN
from sklearn import metrics
import joblib
import pickle

# **Import the data**

from sqlalchemy import create_engine
from urllib.parse import quote
insurance = pd.read_csv(r"AutoInsurance.csv")

# Credentials to connect to Database
user = 'root'  # user name
pw = 'dba@123#'  # password
db = 'amerdb'  # database name
engine = create_engine(f"mysql+pymysql://{user}:%s@localhost/{db}" % quote (f'{pw}'))

# to_sql() - function to push the dataframe onto a SQL table.

insurance.to_sql('ins_tbl', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

sql = 'select * from ins_tbl;'
df = pd.read_sql_query(sql, engine)

# Data types
df.info()
# Drop the unwanted features
df1 = df.drop(['Customer','Effective To Date', 'Location Code'], axis = 1)
# # EXPLORATORY DATA ANALYSIS (EDA) / DESCRIPTIVE STATISTICS

# ***Descriptive Statistics and Data Distribution Function***
df1.describe()
# ## We have to check unique values for categorical data 

df1.State.value_counts()

df1.Response.value_counts()

df1.Education.value_counts()

df1.EmploymentStatus.value_counts()

df1.Gender.value_counts()

df1.Coverage.value_counts()

df1['Marital Status'].value_counts()
df1['Policy Type'].value_counts()
df1['Policy'].value_counts()
df1['Renew Offer Type'].value_counts()
df1['Sales Channel'].value_counts()
df1['Vehicle Class'].value_counts()
df1['Vehicle Size'].value_counts()

# AutoEDA
# Automated Libraries
# pip install dtale
import dtale
d = dtale.show(df)
d.open_browser()
# Missing Data
# Checking Null Values
df1.isnull().sum() # data do not have any null values

# Segregate Numeric and Non-numeric columns
df1.info()

# **By using Mean imputation null values can be impute**
numeric_features = df1.select_dtypes(exclude = ['object']).columns
numeric_features

# Non-numeric columns
categorical_features = df1.select_dtypes(include = ['object']).columns
categorical_features 

# Define Pipeline to deal with Missing data and scaling numeric columns
num_pipeline = Pipeline([('impute', SimpleImputer(strategy = 'mean')), ('scale', MinMaxScaler())])
num_pipeline

# Encoding Non-numeric fields
# **Convert Categorical data  to Numerical data using OneHotEncoder**

categ_pipeline = Pipeline([('OnehotEncode', OneHotEncoder(sparse_output = False))])
categ_pipeline

# Using ColumnTransfer to transform the columns of an array or pandas DataFrame. 
# This estimator allows different columns or column subsets of the input to be
# transformed separately and the features generated by each transformer will
# be concatenated to form a single feature space.
preprocess_pipeline = ColumnTransformer([('categorical', categ_pipeline, categorical_features), 
                                       ('numerical', num_pipeline, numeric_features)], 
                                        remainder = 'passthrough') # Skips the transformations for remaining columns

preprocess_pipeline

# Pass the raw data through pipeline
processed1= preprocess_pipeline.fit(df1) 

# ## Save the Imputation and Encoding pipeline
# import joblib
joblib.dump(processed1, 'processed1')

# File gets saved under current working directory
import os
os.getcwd()

# Clean and processed data for Clustering
ins_clean = pd.DataFrame(processed1.transform(df1), columns = processed1.get_feature_names_out())

ins_clean.describe()

ins_clean.iloc[:,-8:].columns

winsor = Winsorizer(capping_method = 'iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 1.5,
                          variables = ['numerical__Customer Lifetime Value', 'numerical__Income',
       'numerical__Monthly Premium Auto', 'numerical__Months Since Last Claim',
       'numerical__Months Since Policy Inception', 'numerical__Number of Policies',
       'numerical__Total Claim Amount'] )

outlier = winsor.fit(ins_clean[['numerical__Customer Lifetime Value', 'numerical__Income',
       'numerical__Monthly Premium Auto', 'numerical__Months Since Last Claim',
       'numerical__Months Since Policy Inception', 'numerical__Number of Policies',
       'numerical__Total Claim Amount']])

# Save the winsorizer model 
joblib.dump(outlier, 'winsor')

ins_clean[['numerical__Customer Lifetime Value', 'numerical__Income',
       'numerical__Monthly Premium Auto', 'numerical__Months Since Last Claim',
       'numerical__Months Since Policy Inception', 'numerical__Number of Policies',
       'numerical__Total Claim Amount']] = outlier.transform(ins_clean[['numerical__Customer Lifetime Value', 'numerical__Income',
       'numerical__Monthly Premium Auto', 'numerical__Months Since Last Claim',
       'numerical__Months Since Policy Inception', 'numerical__Number of Policies',
       'numerical__Total Claim Amount']])

# # CLUSTERING MODEL BUILDING

# ### KMeans Clustering
# Libraries for creating scree plot or elbow curve 
#from sklearn.cluster import KMeans
#import matplotlib.pyplot as plt

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 12))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(ins_clean)
    TWSS.append(kmeans.inertia_)
TWSS


# ## Creating a scree plot to find out no.of cluster
plt.plot(k, TWSS, 'ro-'); plt.xlabel("No_of_Clusters"); plt.ylabel("total_within_SS")



# ## Using KneeLocator
List = []

for k in range(2, 12):
    kmeans = KMeans(n_clusters = k, init = "k-means++", max_iter = 30, n_init = 10) 
    kmeans.fit(ins_clean)
    List.append(kmeans.inertia_)

#!pip install kneed
from kneed import KneeLocator
kl = KneeLocator(range(2, 12), List, curve = 'convex', direction='decreasing', interp_method='interp1d')
# kl = KneeLocator(range(2, 9), List, curve='convex', direction = 'decreasing')
kl.elbow
plt.style.use("seaborn")
plt.plot(range(2, 12), List)
plt.xticks(range(2, 12))
plt.ylabel("Interia")
plt.axvline(x = kl.elbow, color = 'r', label = 'axvline - full height', ls = '--')
plt.show() 

# From the Kneed locator we can conclude cluster = 4

# Building KMeans clustering
model = KMeans(n_clusters = 4)
yy = model.fit(ins_clean)

# Cluster labels
model.labels_

# ## Cluster Evaluation

# **Silhouette coefficient:**  
# Silhouette coefficient is a Metric, which is used for calculating 
# goodness of clustering technique and the value ranges between (-1 to +1).
# It tells how similar an object is to its own cluster (cohesion) compared to 
# other clusters (separation).
# A score of 1 denotes the best meaning that the data point is very compact 
# within the cluster to which it belongs and far away from the other clusters.
# Values near 0 denote overlapping clusters.

# from sklearn import metrics
metrics.silhouette_score(ins_clean, model.labels_)

# **Calinski Harabasz:**
# Higher value of CH index means cluster are well separated.
# There is no thumb rule which is acceptable cut-off value.
metrics.calinski_harabasz_score(ins_clean, model.labels_)

# **Davies-Bouldin Index:**
# Unlike the previous two metrics, this score measures the similarity of clusters. 
# The lower the score the better the separation between your clusters. 
# Vales can range from zero and infinity
metrics.davies_bouldin_score(ins_clean, model.labels_)

# ### Evaluation of Number of Clusters using Silhouette Coefficient Technique
from sklearn.metrics import silhouette_score

silhouette_coefficients = []

for k in range (2, 12):
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(ins_clean)
    score = silhouette_score(ins_clean, kmeans.labels_)
    k = k
    Sil_coff = score
    silhouette_coefficients.append([k, Sil_coff])

silhouette_coefficients

sorted(silhouette_coefficients, reverse = True, key = lambda x: x[1])


# silhouette coefficients shows the number of clusters 'k = 8' as the best value

# Building KMeans clustering
bestmodel = KMeans(n_clusters = 4)
result = bestmodel.fit(ins_clean)

# ## Save the KMeans Clustering Model
# import pickle
pickle.dump(result, open('Clust_ins.pkl', 'wb'))


# Cluster labels
bestmodel.labels_

mb = pd.Series(bestmodel.labels_) 

# Concate the Results with data
df_clust = pd.concat([mb, df.Customer, df1], axis = 1)
df_clust = df_clust.rename(columns = {0:'cluster_id'})
df_clust.head()

# Aggregate using the mean of each cluster
cluster_agg = df_clust.iloc[:, 3:].groupby(df_clust.cluster_id).mean()
cluster_agg

# Save the Results to a CSV file
# df_clust.to_csv('insurance.csv', encoding = 'utf-8', index = False)


#### DBScan ####
# eps= 2.3, min_samples=7, algorithm='auto', leaf_size=30, 
# metric = 'euclidean', metric_params = None, n_jobs = None, p = None
# eps=9.7, min_samples=2, algorithm='ball_tree', metric='minkowski', leaf_size=90, p=2
 
clustering = DBSCAN(eps = 2.3, min_samples = 7, algorithm = 'auto', 
                    leaf_size = 30, metric = 'euclidean', metric_params = None,
                    n_jobs = None, p = None).fit(ins_clean)

DBSCAN_dataset = df1.copy()
DBSCAN_dataset.loc[:,'Cluster'] = clustering.labels_ 
DBSCAN_dataset.Cluster.value_counts().to_frame()


db_param_options = [[2, 2], [2, 5], [3, 5], [2, 7], [3, 7], [3,10]]

for ep, min_sample in db_param_options:
    db = DBSCAN(eps = ep, min_samples = min_sample)
    db_clusters = db.fit_predict(ins_clean)
    print("Eps: ", ep, "Min Samples: ", min_sample)
    print("DBSCAN Clustering: ", silhouette_score(ins_clean, db_clusters))

pickle.dump(clustering, open('db.pkl', 'wb'))
















