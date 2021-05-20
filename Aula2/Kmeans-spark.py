import numpy as np

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans as KM
from sklearn.metrics import silhouette_score

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession

import pandas as pd
import datetime

spark = SparkSession.builder.appName('SparkKmeans').getOrCreate()

df2 = spark.read.load("/home/silvio/dataset/minute_weather.csv",
                     format="csv", sep=",", inferSchema="true", header="true")
                     
df = df2.drop("rowID","hpwren_timestamp")

df = df.fillna(0)

B=datetime.datetime.now()

cost = []
vecAssembler = VectorAssembler(inputCols=df.columns, outputCol="features")
vector_df = vecAssembler.transform(df)
    
K = range(2,10)
for k in K:
    #kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol('features')
    #model = kmeans.fit(vector_df)
    kmeans = KMeans().setK(k).setSeed(1)
    model = kmeans.fit(vector_df )
    cost.append(model.summary.trainingCost)

E=datetime.datetime.now()
print(E-B)
print(cost)
