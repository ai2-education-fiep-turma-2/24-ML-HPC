import datetime
from pyspark.sql import SparkSession
from pyspark.sql import functions as sf
import pyspark.sql.functions as F
from pyspark.sql.functions import trim, isnan, isnull, when, count, col

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.ml.feature import StringIndexer

from pyspark.ml.feature import Word2Vec


spark = SparkSession.builder.appName("stackoverflow").config("spark.driver.memory", "50g").config("spark.executer.memory", "50g").config("spark.driver.maxResultSize", "50g").getOrCreate()


df_raw2 = spark.read.csv("/home/silvio/dataset/stackoverflow/train.csv", escape='"',multiLine=True,sep=',', ) #, escape='"' )
df_raw2.head()

df_raw2=df_raw2.filter(df_raw2['_c14'] != 'OpenStatus\r')

indexer=StringIndexer(inputCol='_c14',outputCol='OpenStatus_cat')
indexed=indexer.fit(df_raw2).transform(df_raw2)

df_raw33 = indexed.fillna({'_c8':' '})
df_raw34 = df_raw33.fillna({'_c9':' '})
df_raw35 = df_raw34.fillna({'_c10':' '})
df_raw36 = df_raw35.fillna({'_c11':' '})
df_raw37 = df_raw36.fillna({'_c12':' '})

df_raw4 = df_raw37.withColumn('text', sf.concat(sf.col('_c6'),sf.lit(' '), sf.col('_c7')
                                               ,sf.lit(' '), sf.col('_c8')
                                               ,sf.lit(' '), sf.col('_c9')
                                               ,sf.lit(' '), sf.col('_c10')
                                               ,sf.lit(' '), sf.col('_c11')
                                               ,sf.lit(' '), sf.col('_c12')
                                              ))

df_raw5 = df_raw4.withColumn("text", trim(df_raw4.text))

df_raw8 = df_raw5.select("text","OpenStatus_cat")

df_raw8 = df_raw8.withColumn("new_text", F.array(F.col("text")))


#df_raw8.write.parquet("/home/silvio/stackOverflow-prep.parquet")

'''
B=datetime.datetime.now()


word2Vec = Word2Vec(vectorSize=100, minCount=0, inputCol="new_text", outputCol="result")
model = word2Vec.fit(df_raw8)
result = model.transform(df_raw8)

E=datetime.datetime.now()
print(E-B)

#for feature in result.select("result").take(3):
#    print(feature)    


# In[14]:


result.show()


# In[15]:


resultF=result.select("result","OpenStatus_cat")
resultF.show()


# In[16]:


final_data=resultF.select('result','OpenStatus_cat')
train_data,test_data=final_data.randomSplit([0.7,0.3])
train_data.describe().show()


# In[17]:





dt = DecisionTreeClassifier(labelCol="OpenStatus_cat", featuresCol="result")

pipeline = Pipeline(stages=[dt])

model = pipeline.fit(train_data)


# In[18]:


predictions = model.transform(test_data)

predictions.select("prediction", "OpenStatus_cat", "result").show(5)

evaluator = MulticlassClassificationEvaluator(
    labelCol="OpenStatus_cat", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g " % (1.0 - accuracy))

treeModel = model.stages[0]

'''