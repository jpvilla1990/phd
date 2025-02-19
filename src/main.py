import time
import numpy as np
import pandas as pd
from gluonts.model.forecast import SampleForecast
from datasets.datasets import Datasets
from model.moiraiMoe import MoiraiMoE
from evaluation.evaluation import Evaluation

CONTEXT : int = 128
PREDICTION : int = 16
NUMBER_SAMPLES : int = 100
DATASET : str = "electricityUCI"

evaluation : Evaluation = Evaluation()

report : dict = evaluation.evaluateMoiraiMoE(
    CONTEXT,
    PREDICTION,
    NUMBER_SAMPLES,
    DATASET,
)

print(report)

raise Exception("finished")

dataset : Datasets = Datasets()
model : MoiraiMoE = MoiraiMoE(
    predictionLength = PREDICTION,
    contextLenght = CONTEXT,
    numSamples = NUMBER_SAMPLES,
)

iterator = dataset.loadDataset(DATASET)
features = list(iterator.getAvailableFeatures(SUBDATASET).keys())[0:2]
iterator.setSampleSize(CONTEXT + PREDICTION)
iterator.resetIteration(SUBDATASET, True)
while True:
    sample : pd.core.frame.DataFrame = iterator.iterateDataset(SUBDATASET, features)
    pred : SampleForecast = model.inference(sample, DATASET)
    model.plotSample(
        sample.iloc[:CONTEXT],
        sample.iloc[CONTEXT:CONTEXT+PREDICTION],
        DATASET,
    )
    if sample is None:
        break

iterator.setSampleSize(16)
iterator.getAvailableFeatures("electricity")
startTime = time.perf_counter()
samples : list = iterator.loadSample("electricity",1 , 20, ["MT_002", "MT_231"])

iteration : int = 0
iterator.resetIteration("electricity", True)
while iteration < 10:
    iteration += 1
    print(iterator.iterateDataset(
        "electricity",
        ["MT_002", "MT_231"],
    ))

print(f"Elapsed time: {time.perf_counter() - startTime:.6f} seconds")