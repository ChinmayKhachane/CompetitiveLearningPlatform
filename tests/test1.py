from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.ml.feature import MinHashLSH
from pyspark.sql.functions import col
from pyspark.ml.linalg import Vectors
import numpy as np
import pandas as pd
import pyspark
from pyspark.sql import SparkSession
import time

start_t = time.time()
spark = SparkSession.builder.appName('cf').getOrCreate()
schema = StructType() \
    .add("userId", IntegerType(), True) \
    .add("movieId", IntegerType(), True) \
    .add("rate", IntegerType(), True) \
    .add("time", StringType(), True)

df2 = spark.read.schema(schema).csv(
    'train.dat', sep='\t')

ddf = df2.toPandas()
testdata = pd.pivot_table(ddf, values='rate', index=[
                          'movieId'], columns=['userId'], fill_value=0)

item = df2.rdd.map(lambda x: (x[1], (x[0], x[2])))
item = item.groupByKey().mapValues(list)
df2 = item.toDF(["item", "user_rate"])
df3 = df2.toPandas()
item_d = dict(zip(df3.item, df3.index))

ur = df3['user_rate']
user = []
rate = []
for row in ur:
    row.sort(key=lambda x: x[0])
    arr1, arr2 = zip(*row)
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    arr1 = arr1 - 57
    user.append(arr1)
    rate.append(arr2)

df4 = pd.DataFrame({"user": user, "rate": rate})
df4
df3['user'] = df4['user']
df3['rate'] = df4['rate']
df3 = df3.drop(columns=['user_rate'])

df3 = df3.set_index('item')

data = []


def createModel(dataframe, numHashTables):
    for i in dataframe.index:
        tuples = (i, Vectors.sparse(
            943, dataframe['user'][i], dataframe['rate'][i]))
        # data.append([i, Vectors.dense(ndf.loc[i].to_numpy())])
        data.append(tuples)

    psdf = spark.createDataFrame(data, ["id", "features"])

    mh = MinHashLSH(inputCol="features", outputCol="hashes",
                    numHashTables=numHashTables)
    model = mh.fit(psdf)
    return model, psdf


model, psdf = createModel(df3, 6)
end_t = time.time()

print("training time: ", end_t - start_t)

start_p = time.time()


def findKNearN(movie_key, user_key, k=10, model=model, psdf=psdf):
    if movie_key not in df3.index:
        return df3['rate'].loc[movie_key].mean()
    key = Vectors.sparse(943, df3['user'][movie_key], df3['rate'][movie_key])

    kNearN = model.approxNearestNeighbors(psdf, key, k)
    index = 0
    rate_sum = 0
    fsim_sum = 0

    for row in kNearN.collect():
        if index == 0:
            index += 1
            continue
        jaccd_sim = 1 - row['distCol']
        if row['features'][user_key] != 0.0 and jaccd_sim > 0:
            rate_sum += jaccd_sim*row['features'][user_key]
            fsim_sum += jaccd_sim
    if fsim_sum != 0:
        pred = rate_sum / fsim_sum
    else:
        return df3['rate'].loc[movie_key].mean()
    return pred


test_df = spark.read.schema(schema).csv(
    'test.dat', sep='\t')
test = test_df.collect()
pred_rate = []
len = test_df.count()
for i in range(len):
    if test[i][1] == 35 or test[i][1] == 375 or test[i][1] == 30:
        rating = testdata.loc[:, test[i][0]].sum(
        ) / (testdata.loc[:, test[i][0]] != 0).sum()
        pred_rate.append(rating)
        continue
    user_id = test[i][0]
    item_key = test[i][1]
    pred = findKNearN(item_key, user_id - 57, 10)
    pred_rate.append(pred)

end_p = time.time()
print("prediction time:", end_p - start_p)
with open('format_lsh_6_10.txt', 'w') as fp:
    for rate in pred_rate:
        # write each item on a new line
        fp.write("%s\n" % rate)
    print('Done')
