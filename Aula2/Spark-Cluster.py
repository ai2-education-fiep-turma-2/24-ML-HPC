#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# # criando sessão com Spark
# 
# * Sessão pode ser local
# ```
#     spark = SparkSession.builder.appName("stackoverflow").getOrCreate()
#     ```
#     
# * Pode ser submetendo job para um cluster
# 
#     ```
#     spark = SparkSession.builder.appName("stackoverflow").master("spark://10.46.0.12:8893").config("spark.driver.memory","60G").config("spark.executer.memory", "100g").getOrCreate()
#     
#     ```
# * Alguns requisitos podem ser definidos na sessão (ex)
#     * spark.driver.memory
#     * spark.executer.memory
# 

# In[2]:


#spark = SparkSession.builder.appName("stackoverflow").getOrCreate()


# In[3]:


spark = SparkSession.builder.appName("stackoverflow").config("spark.driver.memory", "50g").config("spark.executer.memory", "50g").config("spark.driver.maxResultSize", "50g").getOrCreate()


# In[ ]:


#spark = SparkSession.builder.appName("stackoverflow").master("local[5]").config("spark.driver.memory","60G").getOrCreate()


# In[ ]:


#spark = SparkSession.builder.appName("stackoverflow").master("spark://10.46.0.12:8893").config("spark.driver.memory","60G").config("spark.executer.memory", "100g").getOrCreate()


# # Carregando Dataframe:
#     * Operação ocorre de modo paralelizado, com base em recursos de Dataset e RDD do Spark de modo transparente

# In[4]:


import datetime
'''
B=datetime.datetime.now()


df_raw2 = spark.read.csv("/home/silvio/dataset/stackoverflow/train.csv", escape='"',multiLine=True,sep=',', ) #, escape='"' )
df_raw2.head()

E=datetime.datetime.now()
print(E-B)


# In[5]:


df_raw2=df_raw2.filter(df_raw2['_c14'] != 'OpenStatus\r')

# # comando explain
# * Mostra organização interna so Spark para um objeto

# In[ ]:


df_raw2.explain()


# # Embora seja otimizado, operações em dataframes não podem ser realizadas com comandos SQL, e algumas operações não são paralelizadas
# 
# * As duas operações abaixo levam cerca de 30 segundos e rodam em um recurso

# In[ ]:


import datetime

B=datetime.datetime.now()
print(df_raw2.select("_c2").distinct().count())
print((df_raw2.count(), len(df_raw2.columns)))
E=datetime.datetime.now()
print(E-B)
'''

# # Uma forma de otimizar operações em Dataframes é utilizar arquivos do tipo parquet que são visões do dados otimizada para acesso
# * Escrevendo arquivo parquet

# In[6]:


#df_raw2.write.parquet("/home/silvio/stackOverflow2222.parquet")


# * Lendo arquivo parquet

# In[7]:


parquetFile = spark.read.parquet("/home/silvio/stackOverflow2222.parquet")

parquetFile.createOrReplaceTempView("parquetFile")


# In[ ]:


print(type(parquetFile))
#print(type(df_raw2))


# # parquet aceita qualquer operação SQL

# In[ ]:


#t = spark.sql("SELECT count(_c2) FROM parquetFile")
#t.show()


# In[ ]:


#t = spark.sql("SELECT count(distinct(_c2)) FROM parquetFile")
#t.show()


# * Parquet aceita as operações de dataframe também que são executadas de modo otimizado

# In[ ]:


print(parquetFile.select("_c2").distinct().count())


# ## Arquivos Parquet também são interoperáveis com Pandas (PyArrow)
# * Pré-processamento pode ser feito em Spark com desempenho e repassado ao pandas para facilidade de uso

# # Base Stack overflow
# * Diversas colunas caracterizando perguntas submetidas ao stackoverflow ( mais de 3 milhões de perguntas)
# * Desafio consiste em usar o texto das questões para predizer quais serão encerradas
# 

# # Pre-processamento consiste em:
# * Eliminar valores nulos
# * Criar uma coluna texto com todo texto associado a questão
# * Vetorizar

# In[ ]:


print(parquetFile.select("_c6").head(3))


# In[ ]:


print(parquetFile.select("_c6").head(3))


# In[ ]:


print(parquetFile.select("_c7").head(3))


# In[ ]:


print(parquetFile.select("_c14").head(3))


# In[ ]:


print(parquetFile.select("_c12").tail(13))


# # transformando alvo em categoria ( a partir do parquet para ser mais rápido)

# In[8]:


B=datetime.datetime.now()

indexer=StringIndexer(inputCol='_c14',outputCol='OpenStatus_cat')
indexed=indexer.fit(parquetFile).transform(parquetFile)

E=datetime.datetime.now()
print(E-B)


# In[ ]:





# ## verificando valores nuloes

# In[ ]:

'''
B=datetime.datetime.now()

df_raw3 = df_raw2.select("_c6","_c7","_c8","_c9","_c10","_c11","_c12","_c14")
df_raw3.select([count(when(isnull(c), c)).alias(c) for c in df_raw3.columns]).show()

E=datetime.datetime.now()
print(E-B)
'''

# # eliminado campos nulos

# In[9]:


B=datetime.datetime.now()

df_raw33 = indexed.fillna({'_c8':' '})
df_raw34 = df_raw33.fillna({'_c9':' '})
df_raw35 = df_raw34.fillna({'_c10':' '})
df_raw36 = df_raw35.fillna({'_c11':' '})
df_raw37 = df_raw36.fillna({'_c12':' '})

E=datetime.datetime.now()
print(E-B)


# In[ ]:


#df_raw37.select([count(when(isnull(c), c)).alias(c) for c in df_raw37.columns]).show()


# In[ ]:


#print((df_raw37.count(), len(df_raw37.columns)))


# ## unificando todos os campos de texto em um único chamado text

# In[10]:


df_raw4 = df_raw37.withColumn('text', sf.concat(sf.col('_c6'),sf.lit(' '), sf.col('_c7')
                                               ,sf.lit(' '), sf.col('_c8')
                                               ,sf.lit(' '), sf.col('_c9')
                                               ,sf.lit(' '), sf.col('_c10')
                                               ,sf.lit(' '), sf.col('_c11')
                                               ,sf.lit(' '), sf.col('_c12')
                                              ))

df_raw5 = df_raw4.withColumn("text", trim(df_raw4.text))


# ## selecionando apenas campos de interesse

# In[11]:


df_raw8 = df_raw5.select("text","OpenStatus_cat")


# In[ ]:


#print((df_raw8.count(), len(df_raw8.columns)))


# In[ ]:


#df_raw8.show()


# In[ ]:





# In[12]:



df_raw8 = df_raw8.withColumn("new_text", F.array(F.col("text")))


# In[ ]:


#df_raw8.show()


# In[ ]:


df_raw8.explain()


# ## vetorizando texto

# In[13]:


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

#print(treeModel)


# In[ ]:





# In[ ]:




