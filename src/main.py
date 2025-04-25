import pandas as pd
import numpy as np
from datasetsModule.datasets import Datasets
from model.moiraiMoe import MoiraiMoE
from utils.utils import Utils

CONTEXT : int = 128
PREDICTION : int = 16
NUMBER_SAMPLES : int = 100
DATASET : str = "electricityUCI"
SUBDATASET : str = "electricity"
COLLECTION : str = "moiraiMoECosine_128_16"

from vectorDB.vectorDBingestion import VectorDBIngestion

vectorDBingestion : VectorDBIngestion = VectorDBIngestion()
vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoECosine_128_16")

dataset : Datasets = Datasets()
model : MoiraiMoE = MoiraiMoE(
    predictionLength = PREDICTION,
    contextLength = CONTEXT,
    numSamples = NUMBER_SAMPLES,
)

model.setRagCollection(COLLECTION, DATASET)

iterator = dataset.loadDataset(DATASET)
features = list(iterator.getAvailableFeatures(SUBDATASET).keys())[0:2] # To effectively predict the model requires the first column is the timestamp
iterator.setSampleSize(CONTEXT + PREDICTION)
iterator.resetIteration(SUBDATASET, True)
while True:
    sample : pd.core.frame.DataFrame = iterator.iterateDataset(SUBDATASET, features)
    sample.columns = ["datetime", "value"]
    #pred : np.ndarray = model.inference(sample, DATASET)
    pred : np.ndarray = model.ragInference(sample.iloc[:CONTEXT], DATASET, True, True, plot=True)
    Utils.plot(
        [
            sample["value"].tolist(),
        ],
        "sample.png",
        "-",
        CONTEXT,
    )
    exit()
    if sample is None: # Iterate all samples until the iterator returns empty
        break