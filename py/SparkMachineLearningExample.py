#!/usr/bin/env python
# coding: utf-8

# <h1>Logistic Regression using spark MLlib<h1>

# 
# The dataset is the <strong>Pima Indians Diabetes Dataset</strong> [https://www.kaggle.com/uciml/pima-indians-diabetes-database/version/1]. 
# This dataset belongs to National Institute of Diabetes and Digestive and Kidney Diseases. The dataset contains data used to classifiy if
# someone has diabetes or not. The dataset contains felmale person's data. Features include: number of pregnancies, Glucose concenteration, Blood pressure, skinThickness, 
# Insulin, BMI, DiabetesPedigreeFunction,Age. The label is 1 for diabetic and zero for non diabetic. 268 Participants out of 768 are 1(diabetec). 

# In[1]:


from pyspark.sql import SparkSession #import spark session
spark= SparkSession.builder.appName("MLlib demo").getOrCreate() #Create a spark session using MLlib demo as name


# In[3]:


#conda install -c conda-forge pyspark


# <h3>Load Diabetes Datast stored</h3>

# In[4]:


#Load diabetes.csv from loca storage to a data frame using spark, header=True <The first row is column header>, inferSchema=True, 
#option to inferSchema directly from the dataset
diab_ds=spark.read.csv('diabetes.csv', header=True, inferSchema=True)


# <strong>Get the name of features of the dataset</strong>

# In[5]:


diab_ds.columns


# <strong>Get the datatype of the features</strong>

# In[6]:


diab_ds.dtypes


# <strong> Read First row of dataset</strong>

# In[7]:


diab_ds.head() #returns the first row as key and value


# <p> <strong>In order to see the number of examples for each class, we use groupBy outcome and apply count operation. 
# The result shows that we have 268 diabetic and 500 non diabetic examples. This shows that this dataset is unbalanced. </strong></p>

# In[8]:


diab_ds.groupBy('Outcome').count().show()


# <h3>Dataset preparataion and preprocessing for Logistic regression </h3>
# 
# <p>We need <strong>Vector Assembler </strong> which is a transformer that combines a set of selected features in to a
# a single feature vector. For example for our dataset with 8 features, it will combine these 8 features in to one 
# feature vector. </p>
# 

# In[9]:


#import vectors and vector assembler
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler


# In[10]:


#create vectorAssembler transformer that takes all features and maps them in to one vector called <features>
assembler= VectorAssembler(
inputCols=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction',
           'Age'],
    outputCol="features"

)


# In[11]:


diab_ds_vec=assembler.transform(diab_ds) #Transform diab_db using assembler


# In[12]:


diab_ds_vec.show(3) #show the first element 


# <strong> Now we select features and outcome to build our dataset ready for preprocessing </strong>

# In[13]:


diab_db_final= diab_ds_vec.select('features','Outcome') #select features and Outcome from diab_ds_vec
diab_db_final.show(3) #show first three elements 


# <strong>Feature scaling</strong>
# <p>Since the values of each feture are in different scale, we apply scaling to put all featues on the same scale. We use <strong>Standard Scaler</strong> which is MLlib transformer that transformes a dataset of vector rows by scaling each feature to have a zero mean or unit standard devation. Feature scaling can imporve accuracy for some classifiers</p>

# In[14]:


from pyspark.ml.feature import StandardScaler


# In[15]:


#withStd converts features to unit standard deviation and withMean: centers the data withMean before scaling
scaler=StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)


# In[16]:


#Now build scaler model using diab_db_final
ScalerModel=scaler.fit(diab_db_final)


# In[17]:


#Transform diab_db_final using the scaler model
diab_db_scaled= ScalerModel.transform(diab_db_final)
diab_db_scaled.show(1) #show first row of scaled dataset


# <strong>Now we split our scaled dataset in to training and testing data. We use 75% for training and 25% for testing</strong>

# In[18]:


diab_db_train, diab_db_test=diab_db_scaled.select('scaledFeatures','Outcome').randomSplit([0.75,0.25])
diab_db_train.show(3)


# <strong>Logistic Regression classifer</strong>
# <p>Logistic regression is used to solve binary classification problem. Binomial logistic regeression is used for binary classification and multnomial is used for multi-class classification</p>

# In[19]:


from pyspark.ml.classification import LogisticRegression
lrModel= LogisticRegression(maxIter=50, featuresCol='scaledFeatures', labelCol='Outcome') #build a model by specifying the labelCol as Outcome
lrModel=lrModel.fit(diab_db_train) # Train the model using diab_db_train
trainingSummary= lrModel.summary


# <strong>Evaluation of logistic regression using test data</strong>

# In[20]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator #We use this to evaluate all binary classifications
predictions= lrModel.evaluate(diab_db_test) #apply classification on test data
predictions.predictions.show()
#evaluator= BinaryClassificationEvaluator(rawPredictionCol='predection', labelCol="Outcome")
#evaluator.evaluate(predections.predection)


# In[25]:





# In[ ]:





# In[ ]:





# In[ ]:




