import pandas as pd
import numpy as np
from gluonts.model.forecast import SampleForecast
from datasets.datasets import Datasets
from model.moiraiMoe import MoiraiMoE

CONTEXT : int = 128
PREDICTION : int = 16
NUMBER_SAMPLES : int = 100
DATASET : str = "ET"
SUBDATASET : str = "ETTh1"

dataset : Datasets = Datasets()
model : MoiraiMoE = MoiraiMoE(
    predictionLength = PREDICTION,
    contextLenght = CONTEXT,
    numSamples = NUMBER_SAMPLES,
    collectionName="moiraiMoEAllCosine",
)

iterator = dataset.loadDataset(DATASET)
features = list(iterator.getAvailableFeatures(SUBDATASET).keys())[0:2] # To effectively predict the model requires the first column is the timestamp
iterator.setSampleSize(CONTEXT + PREDICTION)
iterator.resetIteration(SUBDATASET, True)
sample : pd.core.frame.DataFrame = iterator.iterateDataset(SUBDATASET, features)
sampleCopy = sample.copy()
model.ingestVector(sample[1].iloc[:CONTEXT].values, sample[1].iloc[CONTEXT:CONTEXT+PREDICTION].values)
queried : np.ndarray = model.queryVector(sampleCopy[1].iloc[:CONTEXT].values, 1)

print("original")
print(sample[1].values)
print("queried")
print(queried)