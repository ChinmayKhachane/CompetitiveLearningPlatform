#Predicitng ratings using Matrix Factorization | Decomposition
import numpy as np
import pandas as pd

from os.path import join
from numpy.linalg import norm

import time

from pyspark.sql import SparkSession
from sklearn.preprocessing import normalize

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import BucketedRandomProjectionLSH,MinHashLSH


spark = SparkSession.builder.appName("LSH").getOrCreate()


def readData(dpath):
    with open(join(dpath,"train.dat"),"r") as fp:
        data = fp.read().replace("\t"," ").splitlines()
        train_data = [list(map(int,row.split(" ")))[:3] for row in data]
    with open(join(dpath,"test.dat"),"r") as fp:
        test_data = [list(map(int,row.split())) for row in fp.readlines()]
    return train_data,test_data

def getData(dpath = "./data/"):
    train_data,test_data = readData(dpath)
    train_data_dict = dict()
    
    train_data_dict["USER"] = []
    train_data_dict["MOVIE"] = []
    train_data_dict["RATINGS"] = []

    for u,m,r in train_data:
        train_data_dict["USER"].append(u)
        train_data_dict["MOVIE"].append(m)
        train_data_dict["RATINGS"].append(r)
    
    pd_tr_dt = pd.DataFrame(train_data_dict)
    pd_tr_dt.MOVIE -= pd_tr_dt["MOVIE"].min()

    ratings = np.ndarray((np.max(train_data_dict["USER"]),np.max(train_data_dict["MOVIE"])))
    ratings[pd_tr_dt["USER"]-1,pd_tr_dt["MOVIE"]-1] = pd_tr_dt["RATINGS"]

    return pd_tr_dt,ratings,test_data

def predict(data,nli,odata,uid,mid,model,top_k = 15):
    _,crow = nli[uid-1]
    out = model.approxNearestNeighbors(data, crow, top_k+1).collect()
    uidx = np.array([out[i]["ID"]+1 for i in range(1,top_k+1)])
    sims = np.array([out[i]["distCol"] for i in range(1,top_k+1)])
    
    sims = np.exp(sims)
    sims = sims/sims.sum()

    sim_usr = odata[odata.USER.isin(uidx)]
    sim_usr_mv = sim_usr[sim_usr["MOVIE"] == mid - odata.MOVIE.min()]
    rt = 0
    sim_sum = 0
    for ud,ratg in zip(sim_usr_mv.USER.values,sim_usr_mv.RATINGS.values):
        i = np.where(uidx == ud)[0][0]
        rt += (sims[i]*ratg)
        sim_sum += sims[i]
    if sim_sum == 0:
        return odata["RATINGS"].mean()
    return rt/sim_sum


t1 = time.time()
pd_tr_dt,ratings,test_data = getData()
nli = []
for i,row in enumerate(ratings):
    nli.append((i,Vectors.dense(row)))

ratings_mat = spark.createDataFrame(nli,["ID","FT"])
brp = BucketedRandomProjectionLSH(inputCol="FT", outputCol="hashes", bucketLength=20.0,numHashTables=20)
model_brp = brp.fit(ratings_mat)
trans_data_brp = model_brp.transform(ratings_mat)

tmovie_len = max(pd_tr_dt["MOVIE"])+1
user_rat_dict = dict()
jdata = []
for x,y,z in zip(pd_tr_dt["USER"],pd_tr_dt["MOVIE"],pd_tr_dt["RATINGS"]):
    if user_rat_dict.get(x,None) == None:
        user_rat_dict[x] = dict()
    user_rat_dict[x][y] = z

for uid in range(max(pd_tr_dt["USER"]+1)):
    rt = user_rat_dict.get(uid,None)
    if rt == None:
        rt = dict()
        rt[0] = pd_tr_dt["RATINGS"].mean()
    jdata.append((uid,Vectors.sparse(tmovie_len,rt)))

jdata = spark.createDataFrame(jdata,["ID","FT"])
        
mh = MinHashLSH(inputCol="FT",outputCol="hashes",numHashTables=15)
model_mh = mh.fit(jdata)
trans_data_mh = model_mh.transform(jdata)
t2 = time.time()
ot = []
o2 = []
for x,y in test_data:
    ot.append(predict(trans_data_brp,nli,pd_tr_dt,x,y,model_brp))
    o2.append(predict(trans_data_mh,nli,pd_tr_dt,x,y,model_mh))
with open("prdLSH.dat","w") as fp:
    for x,y in zip(ot,o2):
        print(x,y)
    fp.writelines([str((v1*0.3+v2*0.7)) + "\n" for v1,v2 in zip(ot,o2)])
t3 = time.time()

print("Building Time:",t2-t1)
print("Prediction Time:",t3-t2)