#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime
from pyspark.sql import SparkSession
from pyspark.sql import functions as sf
from pyspark.sql.functions import trim, isnan, isnull, when, count, col

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler


# In[ ]:


spark = SparkSession.builder.appName("stackoverflow").getOrCreate()


# In[2]:


spark = SparkSession.builder.appName("stackoverflow").config("spark.driver.memory", "70g").config("spark.executer.memory", "70g").getOrCreate()


# In[ ]:





# In[3]:


#df_raw2 = spark.read.csv("/home/silvio/dataset/stackoverflow/train.csv", escape='"',multiLine=True,sep=',', ) #, escape='"' )
df_raw2 = spark.read.csv("/home/silvio/dataset/stackoverflow/train-sample.csv", escape='"',multiLine=True,sep=',', ) #, escape='"' )
df_raw2


# In[4]:


df_raw2.head(1)


# In[5]:


df_raw3=df_raw2.filter(df_raw2['_c14'] != 'OpenStatus\r')


# In[6]:


print(df_raw3.select("_c6").head(3))


# In[7]:


print(df_raw2.select("_c6").head(3))


# In[8]:


print(df_raw2.select("_c7").head(3))


# In[9]:


print(df_raw2.select("_c14").head(3))


# In[10]:


print(df_raw2.select("_c12").tail(13))


# In[11]:


df_raw3 = df_raw2.select("_c6","_c7","_c8","_c9","_c10","_c11","_c12","_c14")
df_raw3.select([count(when(isnull(c), c)).alias(c) for c in df_raw3.columns]).show()


# In[12]:


df_raw33 = df_raw3.fillna({'_c8':' '})
df_raw34 = df_raw33.fillna({'_c9':' '})
df_raw35 = df_raw34.fillna({'_c10':' '})
df_raw36 = df_raw35.fillna({'_c11':' '})
df_raw37 = df_raw36.fillna({'_c12':' '})


# In[13]:


df_raw37.select([count(when(isnull(c), c)).alias(c) for c in df_raw37.columns]).show()


# In[14]:


print((df_raw37.count(), len(df_raw37.columns)))


# In[15]:




df_raw4 = df_raw37.withColumn('text', sf.concat(sf.col('_c6'),sf.lit(' '), sf.col('_c7')
                                               ,sf.lit(' '), sf.col('_c8')
                                               ,sf.lit(' '), sf.col('_c9')
                                               ,sf.lit(' '), sf.col('_c10')
                                               ,sf.lit(' '), sf.col('_c11')
                                               ,sf.lit(' '), sf.col('_c12')
                                              ))
#df_raw4.show()


df_raw5 = df_raw4.withColumn("text", trim(df_raw4.text))

#df_raw6=df_raw5.filter(df_raw5['text'] != '')
#print(df_raw6.filter(df_raw6['text'] == '').count())

#df_raw6=df_raw5.filter(df_raw5['text'] != '')
#print(df_raw6.filter(df_raw6['text'] == '').count())


# In[ ]:





# In[16]:


from pyspark.ml.feature import StringIndexer

indexer=StringIndexer(inputCol='_c14',outputCol='OpenStatus_cat')
indexed=indexer.fit(df_raw5).transform(df_raw5)


# In[17]:


indexed.show()


# In[18]:


df_raw8 = indexed.select("text","OpenStatus_cat")


# In[19]:


print((df_raw8.count(), len(df_raw8.columns)))


# In[20]:


df_raw8.show()


# In[21]:


df_raw8.select([count(when(isnull(c), c)).alias(c) for c in df_raw8.columns]).show()


# In[ ]:





# In[22]:


#documentDF = sqlContext.createDataFrame([
#    ("Hi I heard about Spark".split(" "), ),
#    ("I wish Java could use case classes".split(" "), ),
#    ("Logistic regression models are neat".split(" "), )
#], ["text"])

#print(documentDF.printSchema())
#print(df_raw8.printSchema())


# In[23]:


import pyspark.sql.functions as F
df_raw8 = df_raw8.withColumn("new_text", F.array(F.col("text")))


# In[24]:


df_raw8.show()

'''

# In[25]:


from pyspark.ml.feature import Word2Vec
from pyspark.sql import SQLContext

sqlContext = SQLContext(spark)

word2Vec = Word2Vec(vectorSize=100, minCount=0, inputCol="new_text", outputCol="result")

model = word2Vec.fit(df_raw8)

result = model.transform(df_raw8)

for feature in result.select("result").take(3):
    print(feature)


# In[26]:


result.show()


# In[27]:


resultF=result.select("result","OpenStatus_cat")
resultF.show()


# In[28]:


final_data=resultF.select('result','OpenStatus_cat')
train_data,test_data=final_data.randomSplit([0.7,0.3])
train_data.describe().show()


# In[29]:


from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Train a DecisionTree model.
dt = DecisionTreeClassifier(labelCol="OpenStatus_cat", featuresCol="result")

# Chain indexers and tree in a Pipeline
pipeline = Pipeline(stages=[dt])

# Train model.  This also runs the indexers.
model = pipeline.fit(train_data)


# In[30]:


# Make predictions.
predictions = model.transform(test_data)

# Select example rows to display.
predictions.select("prediction", "OpenStatus_cat", "result").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="OpenStatus_cat", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g " % (1.0 - accuracy))

treeModel = model.stages[0]
# summary only
print(treeModel)


# In[ ]:





# In[ ]:




'''
