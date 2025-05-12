import numpy as np
import torch
from vectorDB.vectorDB import vectorDB, MilvusVectorDB

context : np.ndarray = np.ones((10))
predict : np.ndarray = np.ones((5)) * 2
query : np.ndarray = np.ones((10)) * 0.9

database : vectorDB = vectorDB()

#database.setCollection("testcollection", "testdataset", lambda x : (torch.tensor(x)*2).unsqueeze(0))
#database.ingestTimeseries(
#    context,
#    predict,
#    "testdataset",
#)
#queried = database.queryTimeseries(query, 1)
#print(queried)

milvusDatabase = MilvusVectorDB()
milvusDatabase.setCollection("testcollection", "testdataset",10)