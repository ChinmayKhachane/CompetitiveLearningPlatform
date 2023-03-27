import re
import sys
import math
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.linalg import Vector, Vectors, VectorUDT
from pyspark.ml.feature import BucketedRandomProjectionLSH
# from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import *
conf = SparkConf()
# conf = SparkConf().setAll([('spark.executor.memory', '8g'), ('spark.executor.cores', '1'), ('spark.cores.max', '1'), ('spark.driver.memory','4g')])
spark = SparkSession.builder.appName("Creating Collaborative Filtering").config(conf=conf).getOrCreate() 
sc = spark.sparkContext
schema = StructType().add("UserID",IntegerType(),True).add("ItemID",IntegerType(),True).add("Rating",IntegerType(),True).add("TimestampUnix",StringType(),True)
df = spark.read.option("delimiter", "\t").schema(schema).csv('file:///'+sys.argv[1])
df.printSchema()
df = df.withColumn("Timestamp", from_unixtime(col("TimestampUnix")))
df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]).show()
a= df.select("UserID").distinct().count()
b = df.select("ItemID").distinct().count()
@udf(VectorUDT())
def to_vector(size, index_list, value_list):
    ind, val = zip(*sorted(zip(index_list, value_list)))
    return Vectors.sparse(size, ind, val)
    
_ = spark.udf.register('to_vector', to_vector)
 
user_features = (df
    .groupBy('UserID')
      .agg(
          collect_list('ItemID').alias('item_list'),
          collect_list('Rating').alias('value_list')
          )
      .crossJoin(
          df.groupBy()
              .agg( max(df.ItemID).alias('max_product_id'))
              .withColumn('size', expr('max_product_id+1'))
              .select('size')
          )
      .withColumn('features', expr('to_vector(size, item_list, value_list)'))
      .select('UserID', 'features')
  )
# ##

# test_df = test_df.withColumn("Rating", lit(0))
# test_df.printSchema()
# test_df.show(5, False)

  
# test_features = (test_df
#     .groupBy('UserID')
#       .agg(
#           collect_list('ItemID').alias('item_list'),
#           collect_list('Rating').alias('value_list')
#           )
#       .crossJoin(
#           test_df.groupBy()
#               .agg( max(test_df.ItemID).alias('max_product_id'))
#               .withColumn('size', expr('max_product_id+1'))
#               .select('size')
#           )
#       .withColumn('features', expr('to_vector(size, item_list,value_list)'))
#       .select('UserID', 'features')
#   ) 


user_features.show(5, False)
fitted_lsh = (
  BucketedRandomProjectionLSH(
    inputCol = 'features', 
    outputCol = 'hash', 
    numHashTables = 25, 
    bucketLength = 0.1025
    )
    .fit(user_features)
  )
 
user_features_bucketed = fitted_lsh.transform(user_features)
user_features_bucketed.select('UserID', 'hash').show(5,False)
user_features_bucketed.agg(count(col('UserID')).alias("UserID_count")).orderBy(col("UserID_count").desc()).show(5,False)
number_of_customers = 10
# transformedA = fitted_lsh.transform(user_features).cache()
# transformedB = fitted_lsh.transform(test_features).cache()

# fitted_lsh.approxSimilarityJoin(transformedA, transformedB, 1.5, distCol="EuclideanDistance")\
#     .select(col("datasetA.UserID").alias("UserID_a"),
#             col("datasetB.UserID").alias("UserID_b"),
#             col("EuclideanDistance")).show()
 
# similar_k_users.show(20, False)
####
# def get_top_users( data ):
#   '''the incoming dataset is expected to have the following structure: user_a, user_b, distance''' 
  
#   rows_to_return = 5 # limit to top 5 users
#   min_score = 1 / (1+math.sqrt(2))
  
# #   data.withColumn('sim', lit(1) / (lit(1) + col('distance')))
#   data['sim']= 1/(1+data['distance'])
# #   similar_k_users
    
#   data['similarity']= (data['sim']-min_score)/(1-min_score)
#   ret = data[['user_id', 'paired_user_id', 'similarity']]
  
#   return ret.sort_values(by=['similarity'], ascending=False).iloc[0:rows_to_return]

schema = StructType([ \
    StructField("UserID",StringType(),True), \
    StructField("ItemID",StringType(),True), \
    StructField("Similarity",IntegerType(),True) \
  ])
 
similarity_df = spark.createDataFrame([],schema=schema)
similarity_df.printSchema()
similarity_df.show(truncate=False)


###
# number_of_neighbors = 5
# union_df = None
# for r in user_features_bucketed.select(col("UserID"), col("features")).toLocalIterator():
#     r_dict = r.asDict()
#     user_features = r_dict['features']
#     user_id = r_dict['UserID']
#     res = (
#       fitted_lsh.approxNearestNeighbors(user_features_bucketed, user_features, number_of_neighbors)
#        .withColumn("UserID_a", lit(user_id))
#     .select(col("UserID_a"), col("UserID").alias("UserID_b"), col("distCol")))
#     if union_df is None: union_df = res
#     else: union_df = res.union(union_df)
# union_df.show(30, False)
# # union_df.to_csv('file://'+sys.argv[3] + '_Result')
# union_df_one = union_df.coalesce(1)
# union_df_one.write.option("header",True).csv('file://'+sys.argv[3] + '_Result1')

# similar_users = spark.read.option("header",True).csv('file://'+sys.argv[3] + '_Result1/part-00000-d8426d87-4052-4c51-9856-9cc4e0917ea7-c000.csv')
# similar_users.show(10,False)


# ## join for features
# users_rating = similar_users.join(user_features,similar_users.UserID_a ==  user_features.UserID,"inner") \
#     .select("UserId_b","UserID", "features") \
#     .show(10, truncate=False)
# ##
# #users_rating.select("features).show(10, truncate=False)
# for r in test_df.select(col("UserID")):
#   print(r)
  # if test_df['UserID'].isin(users_rating['UserID']):


neighbors = spark.read.format("csv").option("header", "true").load('file://'+sys.argv[3] + '_Result1/part-00000-d8426d87-4052-4c51-9856-9cc4e0917ea7-c000.csv')
neighborsSum = neighbors.groupBy(col("UserID_a")).agg(sum(col("distCol")).alias("totalDist")).cache()
joinedNeighbors = neighbors.join(neighborsSum, "UserID_a", "left_outer")
finalNeighbors = joinedNeighbors.withColumn("scaledSimilarity", (joinedNeighbors.totalDist - joinedNeighbors.distCol)/joinedNeighbors.totalDist)
testSchema = StructType().add("UserID",IntegerType(),True).add("ItemID",IntegerType(),True)
testDF = spark.read.option("delimiter", "\t").schema(testSchema).csv('file:///'+sys.argv[2]).cache()
testDF.printSchema()

predictions = []
for r in testDF.limit(10).toLocalIterator():
  r_js = r.asDict()
  userId = r_js['UserID']
  itemId = r_js['ItemID']
  rating = 0.0
  fFinalNeighbors = finalNeighbors.filter(col("UserID_a") == userId)
  fDF = df.filter(col("ItemID") == itemId)
#   fFinalNeighbors.show(5, False)
#   fDF.show(5, False)
  j = fFinalNeighbors.join(fDF, fFinalNeighbors.UserID_b == fDF.UserID, "inner").select(col("scaledSimilarity"), col("Rating")).withColumn("predictedRating", col("scaledSimilarity")*col("Rating"))
  if(j.limit(1).count() > 0 ):
    rating = j.groupBy().agg(avg('predictedRating').alias("avgPredictedRating")).collect()[0]['avgPredictedRating']
  # predictions.append([userId, itemId, rating, __builtin__.round(rating)])
  predictions.append([userId, itemId, rating, math.ceil(rating)])


predictSchema = StructType().add("UserID",IntegerType(),True).add("ItemID",IntegerType(),True).add("rating",DoubleType(),True).add("roundOffRating",IntegerType(),True)
predictionsDF = spark.createDataFrame(predictions, predictSchema).cache()
predictionsDF.show(5,False)
final = predictionsDF.select(col("roundOffRating").alias("Rating"))
finalRating = predictionsDF.select(col("rating").alias("Rating"))
# final.write.text('file://'+sys.argv[3] + 'Rounded_Ratings3.txt')
# finalRating.write.text('file://'+sys.argv[3] + '_Ratings3.txt')

# predictionsDF.coalesce(1).write.format("text").option("header",True).mode("append").save('file://'+sys.argv[3] + 'UserID_Ratings3.txt')
# final.rdd.coalesce(1).write.format("text").option("header",False).mode("append").save('file://'+sys.argv[3] + 'Rounded_Ratings3.txt')
# finalRating.rdd.coalesce(1).write.format("text").option("header",False).mode("append").save('file://'+sys.argv[3] + '_Ratings3.txt')

finalRating.rdd.coalesce(1).saveAsTextFile('file://'+sys.argv[3] + '_Ratings3.txt')
final.rdd.coalesce(1).saveAsTextFile('file://'+sys.argv[3] + 'Rounded_Ratings3.txt')










## function 
# @pandas_udf("UserID int, paired_user_id int, similarity double", PandasUDFType.GROUPED_MAP)
# def get_top_users( data ):  
#   rows_to_return = 5 # limit to top 5 users
#   min_score = 1 / (1+math.sqrt(2))
  
# #   data.withColumn('sim', lit(1) / (lit(1) + col('distance')))
#   data['sim']= 1/(1+data['distance'])
# #   similar_k_users
    
#   data['similarity']= (data['sim']-min_score)/(1-min_score)
#   ret = data[['user_id', 'paired_user_id', 'similarity']]
  
#   return ret.sort_values(by=['similarity'], ascending=False).iloc[0:rows_to_return]


##
# sample_fraction = 0.30
 
# # calculate max possible distance between users
# max_distance = math.sqrt(50)
 
# # calculate min possible similarity (unscaled)
# min_score = 1 / (1 + math.sqrt(2))
 
#perform similarity join for sample of users
# sample_comparisons = (
#   fitted_lsh.approxSimilarityJoin(
#     user_features_bucketed, # use a random sample for our target users
#     user_features_bucketed_b,
#     threshold = max_distance,
#     distCol = 'distance'
#     )
#     .withColumn('similarity', lit(1)/(lit(1)+col('distance')))
#     .withColumn('similarity_rescaled', (col('similarity') - lit(min_score)) / lit(1.0 - min_score))
#     .select(
#     col("datasetA.UserID").alias("UserID_a"),
#     col("datasetB.UserID").alias("UserID_b"),
#     col("similarity_rescaled")).groupBy('UserID_a').agg(count(col('UserID_a')).alias("UserID_count")).orderBy(col("UserID_count").desc())
#     # .groupBy('UserID_a')
#     #     .apply(
#     #       get_top_users,
#     #       schema='''
#     #         UserID int,
#     #         paired_user_id int,
#     #         similarity double
#     #         '''
#     #     )
#     )

# sample_comparisons.show(50, False)


sc.stop()