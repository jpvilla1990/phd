import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import math

from jaxtyping import Bool, Float, Int

from uni2ts.eval_util.plot import plot_single

from gluonts.torch import PyTorchPredictor
import uni2ts
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule
from gluonts.dataset.common import ListDataset
from gluonts.model.forecast import SampleForecast

from utils.timeManager import TimeManager
from utils.fileSystem import FileSystem

from vectorDB.vectorDB import vectorDB
from exceptions.modelException import ModelException

class MoiraiMoEEmbeddings(nn.Module):
    def __init__(self, moiraRaiModule : MoiraiMoEModule):
        super().__init__()
        self.__device : str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__targetDevice : str = "cpu"
        # Extracting layers from the original model including first normalization layer before attention module
        self.scaler : uni2ts.module.packed_scaler.PackedStdScaler = moiraRaiModule.scaler
        self.inProj : uni2ts.module.ts_embed.MultiInSizeLinear = moiraRaiModule.in_proj
        self.resProj : uni2ts.module.ts_embed.MultiInSizeLinear = moiraRaiModule.res_proj
        self.featProj : uni2ts.module.ts_embed.FeatLinear = moiraRaiModule.feat_proj
        self.norm : uni2ts.module.norm.RMSNorm = moiraRaiModule.encoder.layers[0].norm1

    def forward(
            self,
            x : np.ndarray,
            patchSize : int = 16,
            batchSize : int = 1,
        ) -> torch.Tensor:
        """
        Define the forward pass based on the selected layers.
        Assuming:
        - `scaler` normalizes input.
        - `in_proj`, `res_proj`, and `feat_proj` apply transformations.
        """
        seqLen : int = math.ceil(len(x) / patchSize) + 1
        patchSizeTensor : torch.Tensor = torch.full((batchSize, seqLen), patchSize).to(self.__device)
        target : torch.Tensor = F.pad(
            torch.tensor(x, dtype=torch.float32).reshape(batchSize, seqLen - 1, patchSize),
            (0, 0, 0, 1),
            value=0,
        ).to(self.__device)
        observedMask : torch.Tensor = torch.ones((batchSize, seqLen, patchSize), dtype=torch.bool).to(self.__device)
        predictionMask : torch.Tensor = torch.zeros((batchSize, seqLen), dtype=torch.bool).to(self.__device)
        predictionMask[0][:][-1] = True
        sampleId : torch.Tensor = torch.ones((batchSize, seqLen), dtype=torch.int32).to(self.__device)
        variateId : torch.Tensor = torch.zeros((batchSize, seqLen), dtype=torch.int32).to(self.__device)

        loc, scale = self.scaler(
            target,
            observedMask * ~predictionMask.unsqueeze(-1),
            sampleId,
            variateId,
        )
        scaledTarget : torch.Tensor = (target - loc) / scale
        inRepr : torch.Tensor = self.inProj(scaledTarget, patchSizeTensor)
        inRepr : torch.Tensor = F.silu(inRepr)
        inRepr : torch.Tensor = self.featProj(inRepr, patchSizeTensor)
        resRepr : torch.Tensor = self.resProj(scaledTarget, patchSizeTensor)

        # Combine or return outputs depending on your use case
        return self.norm(inRepr + resRepr)

    def inference(
            self,
            x : np.ndarray,
            patchSize : int = 16,
            batchSize : int = 1,
        ) -> torch.Tensor:
        """
        Inference
        """
        output : torch.Tensor = None
        with torch.no_grad():
            output = self(x, patchSize, batchSize)

        return output.to(self.__targetDevice)

class MoiraiMoE(FileSystem):
    """
    Class to handle MoiraiMoe
    """
    def __init__(
        self,
        modelSize : str = "small",
        predictionLength : int = 20,
        contextLength : int = 200,
        patchSize : int = 16,
        numSamples : int = 100,
        targetDim : int = 1,
        featDynamicRealDim : int = 0,
        pastFeatDynamicRealDim : int = 0,
        batchSize : int = 1,
        collectionName : str = "moiraiMoEAllCosine_32_16",
        rag : bool = False
    ):
        super().__init__()
        self.__datasetsConfig : dict = self._getConfig()["datasets"]
        self.__timestampFormat : str = "%d-%m-%Y %H:%M:%S"
        self.__timeDiffsSeconds : dict = {
            "S" : 1,
            "T" : 60,
            "H" : 60 * 60,
            "D" : 60 * 60 * 24,
            "W" : 60 * 60 * 24 * 7,
            "M" : 60 * 60 * 24 * 30,
            "Y" : 60 * 60 * 24 * 365,
        }

        if rag:
            contextLength += contextLength + predictionLength

        self.__contextLength : int = contextLength
        self.__model : MoiraiMoEForecast = MoiraiMoEForecast(
            module=MoiraiMoEModule.from_pretrained(f"Salesforce/moirai-moe-1.0-R-{modelSize}"),
            prediction_length=predictionLength,
            context_length=contextLength,
            patch_size=patchSize,
            num_samples=numSamples,
            target_dim=targetDim,
            feat_dynamic_real_dim=featDynamicRealDim,
            past_feat_dynamic_real_dim=pastFeatDynamicRealDim,
        )

        self.__predictor : PyTorchPredictor = self.__model.create_predictor(batch_size=batchSize)

        self.__moiraiMoEEmbeddings : MoiraiMoEEmbeddings = MoiraiMoEEmbeddings(self.__model.module)
        self.__vectorDB : vectorDB = vectorDB()

    def setRagCollection(self, collectionName : str, dataset : str):
        """
        Method to set RAG collection
        """
        self.__vectorDB.setCollection(
            collectionName,
            dataset,
            self.__moiraiMoEEmbeddings.inference,
        )

    def __getFrequency(self, timestamps : pd.core.frame.DataFrame, timestampFormat : str) -> str:
        """
        Method to get frequency from the timestamps
        """
        timeDiffSeconds : int = TimeManager.timeDiffSeconds(timestamps.iloc[0], timestamps.iloc[1], timestampFormat)

        distances : dict = {abs(self.__timeDiffsSeconds[key] - timeDiffSeconds) : key for key in self.__timeDiffsSeconds}

        return distances[sorted(distances.keys())[0]]

    def ingestVector(self, sample : np.ndarray, prediction : np.ndarray, dataset : str = ""):
        """
        Method to ingest vector
        """
        self.__vectorDB.ingestTimeseries(sample, prediction, dataset)

    def deleteDataset(self, dataset : str):
        """
        Method to delete dataset from collection
        """
        self.__vectorDB.deleteDataset(dataset)

    def queryVector(self, sample : np.ndarray, k : int = 1, metadata : dict = {}) -> tuple:
        """
        Method to query vector
        """
        return self.__vectorDB.queryTimeseries(sample, k, metadata)

    def inference(self, sample : pd.core.frame.DataFrame, dataset : str) -> SampleForecast:
        """
        Method to predict one sample, first columns must be the timestamp and second is the timeseries
        """
        if len(sample.columns) != 2:
            raise ModelException("MoiraiMoE predictor accepts only two columns, timestamp and timeseries itself")

        timestampFormat : str = self.__datasetsConfig[dataset]["timeformat"]

        sample.columns = ["datetime", "value"]
        sampleGluonts : ListDataset = ListDataset(
            [{
                "start": TimeManager.convertTimeFormat(sample["datetime"].iloc[0], timestampFormat, self.__timestampFormat),
                "target": sample["value"].tolist(),
            }],
            freq=self.__getFrequency(sample["datetime"].iloc[0:2], timestampFormat)
        )
        return next(iter(self.__predictor.predict(sampleGluonts)))

    def ragInference(self, sample : pd.core.frame.DataFrame, dataset : str) -> SampleForecast:
        """
        Method to predict one sample, first columns must be the timestamp and second is the timeseries
        """
        if len(sample.columns) != 2:
            raise ModelException("MoiraiMoE predictor accepts only two columns, timestamp and timeseries itself")

        timestampFormat : str = self.__datasetsConfig[dataset]["timeformat"]

        sample.columns = ["datetime", "value"]
        queried : np.ndarray = self.queryVector(sample["value"], k=1, metadata={"dataset" : dataset})[0][0]
        sampleNp : np.ndarray = sample["value"].to_numpy()
        queriedMean, queriedStd = np.mean(queried), np.std(queried)
        sampleMean, sampleStd = np.mean(sampleNp), np.std(sampleNp)

        queryNormed : np.ndarray = (queried - queriedMean) / (queriedStd + 1e-8)
        queryDenormed : np.ndarray = (queryNormed * sampleStd) + sampleMean
        newSample : list = queryDenormed.tolist() + sampleNp.tolist()
        sampleGluonts : ListDataset = ListDataset(
            [{
                "start": TimeManager.convertTimeFormat(sample["datetime"].iloc[0], timestampFormat, self.__timestampFormat),
                "target": newSample,
            }],
            freq=self.__getFrequency(sample["datetime"].iloc[0:2], timestampFormat)
        )
        return next(iter(self.__predictor.predict(sampleGluonts)))

    def plotSample(self, sample : pd.core.frame.DataFrame, groundTruth : pd.core.frame.DataFrame, dataset : str):
        """
        Method to plot sample, first columns must be the timestamp and second is the timeseries
        """
        if len(sample.columns) != 2 or len(groundTruth.columns) != 2:
            raise ModelException("MoiraiMoE predictor accepts only two columns, timestamp and timeseries itself")

        timestampFormat : str = self.__datasetsConfig[dataset]["timeformat"]

        sample.columns = ["datetime", "value"]
        groundTruth.columns = ["datetime", "value"]

        sampleDict : dict = {
            "start": TimeManager.convertTimeFormat(sample["datetime"].iloc[0], timestampFormat, self.__timestampFormat),
            "target": sample["value"].tolist(),
        }
        groundTruthDict : dict = {
            "start": TimeManager.convertTimeFormat(groundTruth["datetime"].iloc[0], timestampFormat, self.__timestampFormat),
            "target": groundTruth["value"].tolist()
        }
        sampleGluonts : ListDataset = ListDataset(
            [sampleDict],
            freq=self.__getFrequency(sample["datetime"].iloc[0:2], timestampFormat),
        )

        prediction : SampleForecast = next(iter(self.__predictor.predict(sampleGluonts)))

        plot_single(
            sampleDict,
            groundTruthDict,
            prediction,
            context_length=self.__contextLength,
            name="pred",
            show_label=True,
        )
        plt.show()