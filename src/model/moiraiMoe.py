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
from utils.utils import Utils

from vectorDB.vectorDB import vectorDB
from exceptions.modelException import ModelException

class MoiraiMoEEmbeddings(nn.Module):
    def __init__(self, moiraRaiModule : MoiraiMoEModule):
        super().__init__()
        self.__device : str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__device = "cpu"
        self.__targetDevice : str = "cpu"
        # Extracting layers from the original model including first normalization layer before attention module
        self.scaler : uni2ts.module.packed_scaler.PackedStdScaler = moiraRaiModule.scaler
        self.inProj : uni2ts.module.ts_embed.MultiInSizeLinear = moiraRaiModule.in_proj.to(self.__device)
        self.resProj : uni2ts.module.ts_embed.MultiInSizeLinear = moiraRaiModule.res_proj.to(self.__device)
        self.featProj : uni2ts.module.ts_embed.FeatLinear = moiraRaiModule.feat_proj.to(self.__device)
        self.norm : uni2ts.module.norm.RMSNorm = moiraRaiModule.encoder.layers[0].norm1.to(self.__device)

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
        pad : np.ndarray = np.zeros((batchSize * (seqLen - 1) * patchSize) - len(x))
        if len(pad) > 0:
            x = np.concatenate([pad, x])
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
        createPredictor : bool = True,
        frozen : bool = True,
    ):
        super().__init__()
        self.__patchSize : int = patchSize
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

        self.__device : str = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.__scoreThreshold : float = self._getConfig()["vectorDatabase"]["scoreThreshold"]
        self.__k : int = self._getConfig()["vectorDatabase"]["k"]

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

        self.__modelRag : MoiraiMoEForecast = MoiraiMoEForecast(
            module=MoiraiMoEModule.from_pretrained(f"Salesforce/moirai-moe-1.0-R-{modelSize}"),
            prediction_length=predictionLength,
            context_length= contextLength + predictionLength,
            patch_size=patchSize,
            num_samples=numSamples,
            target_dim=targetDim,
            feat_dynamic_real_dim=featDynamicRealDim,
            past_feat_dynamic_real_dim=pastFeatDynamicRealDim,
        )

        self.__modelRagCA : MoiraiMoEForecast = MoiraiMoEForecast(
            module=MoiraiMoEModule.from_pretrained(f"Salesforce/moirai-moe-1.0-R-{modelSize}"),
            prediction_length=predictionLength,
            context_length= contextLength + contextLength,
            patch_size=patchSize,
            num_samples=numSamples,
            target_dim=targetDim,
            feat_dynamic_real_dim=featDynamicRealDim,
            past_feat_dynamic_real_dim=pastFeatDynamicRealDim,
        )

        self.__moiraiMoEEmbeddings : MoiraiMoEEmbeddings = MoiraiMoEEmbeddings(self.__model.module)
        self.__vectorDB : vectorDB = vectorDB()

        if createPredictor:
            self.__predictor : PyTorchPredictor = self.__model.create_predictor(batch_size=batchSize)
            self.__predictorRag : PyTorchPredictor = self.__modelRag.create_predictor(batch_size=batchSize)

        if frozen:
            for param in self.__model.module.parameters():
                param.requires_grad = False
            for param in self.__modelRag.module.parameters():
                param.requires_grad = False
            for param in self.__modelRagCA.module.parameters():
                param.requires_grad = False

        self.__model.module = self.__modelRagCA.module.to(self.__device)
        self.__modelRag.module = self.__modelRagCA.module.to(self.__device)
        self.__modelRagCA.module = self.__modelRagCA.module.to(self.__device)

    def __patching(
        self,
        x : torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply patching to the input tensor.
        This method reshapes the input tensor into patches of a specified size.
        :param x: Input tensor to be patched.
        :return: Patches of the input tensor.
        """
        seqLength = x.shape[1]
        remainder : int  = (seqLength - self.__patchSize) % self.__patchSize
        padLen : int = self.__patchSize - remainder if remainder != 0 else 0
        x = F.pad(x, (0, padLen), value=0)
        x = x.unfold(dimension=1, size=self.__patchSize, step=self.__patchSize)
        zeros : torch.Tensor = torch.zeros(x.shape[0], 1, x.shape[2]).to(x.device)
        return torch.cat([x, zeros], dim=1)

    def forwardRagCA(self, x : torch.Tensor) -> torch.Tensor:
        """
        Forward pass RAG Cross Attention
        """
        device : str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        target : torch.Tensor = self.__patching(x)

        batchSize : int = target.shape[0]
        numPatches : int = target.shape[1]
        patchSizeTensor : torch.Tensor = torch.full(
            (batchSize, numPatches),
            self.__patchSize,
            device=device,
        )
        observedMask : torch.Tensor = torch.ones(
            (batchSize, numPatches, self.__patchSize),
            dtype=torch.bool,
            device=device,
        )
        predictionMask : torch.Tensor = torch.zeros(
            (batchSize, numPatches),
            dtype=torch.bool,
            device=device,
        )
        predictionMask[0][:][-1] = True
        sampleId : torch.Tensor = torch.ones(
            (batchSize, numPatches),
            dtype=torch.int32,
            device=device,
        )
        timeId : torch.Tensor = torch.arange(
            numPatches,
            dtype=torch.int32,
            device=device,
        )
        timeId = timeId.unsqueeze(0)
        timeId = timeId.repeat(batchSize, 1)

        variateId : torch.Tensor = torch.zeros(
            (batchSize, numPatches),
            dtype=torch.int32,
            device=device,
        )

        return self.__modelRagCA.module(
            target=target,
            observed_mask=observedMask,
            sample_id=sampleId,
            time_id=timeId,
            variate_id=variateId,
            prediction_mask=predictionMask,
            patch_size=patchSizeTensor,
        )

    def setRagCollection(self, collectionName : str, dataset : str):
        """
        Method to set RAG collection
        """
        self.__vectorDB.setCollection(
            collectionName,
            dataset,
            self.__moiraiMoEEmbeddings.inference,
        )

    def setRafCollection(self, collectionName : str, dataset : str):
        """
        Method to set RAF collection
        """
        self.__vectorDB.setCollection(
            collectionName,
            dataset,
            lambda x : torch.tensor(x).reshape(1, len(x)),
        )

    def setRafCosCollection(self, collectionName : str, dataset : str):
        """
        Method to set RAF collection
        """
        self.__vectorDB.setCollection(
            collectionName,
            dataset,
            lambda x : (torch.tensor(x).reshape(1, len(x)) - torch.mean(torch.tensor(x).reshape(1, len(x)))) / (torch.std(torch.tensor(x).reshape(1, len(x))) + 1e-8),
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

    def deleteCollection(self, collectionName : str, dataset : str):
        """
        Method to delete a collection
        """
        self.__vectorDB.deleteCollection(collectionName, dataset)

    def queryVector(self, sample : np.ndarray, k : int = 1, metadata : dict = {}) -> tuple:
        """
        Method to query vector
        """
        return self.__vectorDB.queryTimeseries(sample, k, metadata)

    def queryBatchVector(self, batch : torch.Tensor, k : int = 1, metadata : dict = {}) -> tuple:
        """
        Method to query vector
        """
        batch = batch.to("cpu")
        queriedBatch : torch.Tensor = None
        scoreBatch : torch.Tensor = None
        for element in batch:
            queried, score = self.__vectorDB.queryTimeseries(element, k, metadata)
            #queriedTorch : torch.Tensor = torch.Tensor(queried).to(self.__device).unsqueeze(0)
            #scoreTensor : torch.Tensor = torch.Tensor(score).to(self.__device).unsqueeze(0)
            #if queriedBatch is None:
            #    queriedBatch = queriedTorch
            #    scoreBatch = scoreTensor
            #else:
            #    torch.cat((queriedBatch, queriedTorch), dim=0)
            #    torch.cat((scoreBatch, scoreTensor), dim=0)
        print(queriedBatch.shape)
        print(scoreBatch.shape)
        print(queried)
        batch = batch.to(self.__device)
        return batch, torch.randn(1, 1, 1).to(batch.device)
        return self.__vectorDB.queryTimeseries(sample, k, metadata)

    def mergeQueries(self, query : tuple) -> np.ndarray:
        """
        Method to merge queries
        """
        merged : np.ndarray = np.zeros_like(query[0][0])
        nElements : int = 0
        for index in range(len(query[0])):
            if query[1][index] > self.__scoreThreshold:
                merged += query[0][index]
                nElements += 1

        if nElements == 0:
            return None
        else:
            return merged / nElements

    def mergeQueriesSoftMax(self, query : tuple, cosine : bool) -> np.ndarray:
        """
        Method to merge queries
        """
        vectors : np.ndarray = np.array(query[0])
        scores : np.ndarray = np.array(query[1]) if cosine else np.array(query[1])/(np.std(np.array(query[1])) + 1e-8)

        validIndices : np.ndarray = scores > self.__scoreThreshold if cosine else scores > 0 # If score is L2, then the score does not makes sense and just filters negative L2

        validScores : np.ndarray = scores[validIndices]
        validVectors : np.ndarray = vectors[validIndices]

        softMaxNumerator : np.ndarray = np.exp(validScores)
        weights : np.ndarray = softMaxNumerator / (softMaxNumerator.sum() + 1e-8)

        weigthedVectors : np.ndarray = validVectors * weights[:, np.newaxis]

        if len(validScores) == 0:
            return None
        else:
            return np.sum(weigthedVectors, axis=0)

    def inference(self, sample : pd.core.frame.DataFrame, dataset : str) -> np.ndarray:
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
        return next(iter(self.__predictor.predict(sampleGluonts))).quantile(0.5)

    def ragInference(
            self,
            sample : pd.core.frame.DataFrame,
            dataset : str,
            softMax : bool = False,
            cosine : bool = True,
            ragPredOnly : bool = False,
            plot : bool = False,
        ) -> np.ndarray:
        """
        Method to predict one sample, first columns must be the timestamp and second is the timeseries
        """
        if len(sample.columns) != 2:
            raise ModelException("MoiraiMoE predictor accepts only two columns, timestamp and timeseries itself")

        timestampFormat : str = self.__datasetsConfig[dataset]["timeformat"]

        sample.columns = ["datetime", "value"]
        queriedVectors : tuple = self.queryVector(sample["value"], k=self.__k, metadata={"dataset" : dataset})
        queried : np.ndarray = self.mergeQueries(queriedVectors) if not softMax else self.mergeQueriesSoftMax(queriedVectors, cosine)
        if queried is not None:
            sampleNp : np.ndarray = sample["value"].to_numpy()
            queriedMean, queriedStd = np.mean(queried), np.std(queried)
            sampleMean, sampleStd = np.mean(sampleNp), np.std(sampleNp)

            queryNormed : np.ndarray = (queried - queriedMean) / (queriedStd + 1e-8)
            sampleNormed : np.ndarray = (sampleNp - sampleMean) / (sampleStd + 1e-8)

            #difference : float = sampleNormed[0] - queryNormed[-1]
            #queryNormed += difference

            newSample : list = queryNormed.tolist() + sampleNormed.tolist()
            sampleGluonts : ListDataset = ListDataset(
                [{
                    "start": TimeManager.convertTimeFormat(sample["datetime"].iloc[0], timestampFormat, self.__timestampFormat),
                    "target": newSample,
                }],
                freq=self.__getFrequency(sample["datetime"].iloc[0:2], timestampFormat)
            )
            predictionNormed : np.ndarray = next(iter(self.__predictorRag.predict(sampleGluonts))).quantile(0.5)
            prediction : np.ndarray = (predictionNormed * (sampleStd + 1e-8)) + sampleMean

            #predictions : np.ndarray = (
            #    [prediction] + [vector[self.__contextLength:] for vector in queriedVectors[0]],
            #    [sum([(2 - vector) / len(queriedVectors[1]) for vector in queriedVectors[1]])] + [vector / len(queriedVectors[1]) for vector in queriedVectors[1]],
            #)
            #prediction = self.mergeQueries(predictions) if not softMax else self.mergeQueriesSoftMax(predictions, cosine)

            if plot:
                Utils.plot(
                    [query.tolist() for query in queriedVectors[0]],
                    "rag.png",
                    ":",
                    self.__contextLength,
                )
                Utils.plot(
                    [queried.tolist()],
                    "ragAvg.png",
                    ":",
                    self.__contextLength,
                )
                Utils.plot(
                    [newSample + predictionNormed.tolist()],
                    "inputAndRag.png",
                    ":",
                    self.__contextLength,
                    rag=True,
                )
                Utils.plot(
                    [sample["value"].tolist() + prediction.tolist()],
                    "pred.png",
                    "-",
                    self.__contextLength,
                )

            return prediction
        else:
            sampleGluonts : ListDataset = ListDataset(
                [{
                    "start": TimeManager.convertTimeFormat(sample["datetime"].iloc[0], timestampFormat, self.__timestampFormat),
                    "target": sample["value"].tolist(),
                }],
                freq=self.__getFrequency(sample["datetime"].iloc[0:2], timestampFormat)
            )
            return next(iter(self.__predictor.predict(sampleGluonts))).quantile(0.5)

    def rafInference(
            self,
            sample : pd.core.frame.DataFrame,
            dataset : str,
            softMax : bool = False,
            cosine : bool = True,
            ragPredOnly : bool = False,
            plot : bool = False,
        ) -> np.ndarray:
        """
        Method to predict one sample, first columns must be the timestamp and second is the timeseries
        """
        if len(sample.columns) != 2:
            raise ModelException("MoiraiMoE predictor accepts only two columns, timestamp and timeseries itself")

        timestampFormat : str = self.__datasetsConfig[dataset]["timeformat"]

        sample.columns = ["datetime", "value"]
        queriedVectors : tuple = self.queryVector(sample["value"], k=self.__k, metadata={"dataset" : dataset})
        queried : np.ndarray = self.mergeQueries(queriedVectors) if not softMax else self.mergeQueriesSoftMax(queriedVectors, cosine)
        if queried is not None:
            sampleNp : np.ndarray = sample["value"].to_numpy()
            queriedMean, queriedStd = np.mean(queried), np.std(queried)
            sampleMean, sampleStd = np.mean(sampleNp), np.std(sampleNp)

            queryNormed : np.ndarray = (queried - queriedMean) / (queriedStd + 1e-8)
            sampleNormed : np.ndarray = (sampleNp - sampleMean) / (sampleStd + 1e-8)

            difference : float = sampleNormed[0] - queryNormed[-1]
            queryNormed += difference

            newSample : list = queryNormed.tolist() + sampleNormed.tolist()
            sampleGluonts : ListDataset = ListDataset(
                [{
                    "start": TimeManager.convertTimeFormat(sample["datetime"].iloc[0], timestampFormat, self.__timestampFormat),
                    "target": newSample,
                }],
                freq=self.__getFrequency(sample["datetime"].iloc[0:2], timestampFormat)
            )
            predictionNormed : np.ndarray = next(iter(self.__predictorRag.predict(sampleGluonts))).quantile(0.5)
            prediction : np.ndarray = (predictionNormed * (sampleStd + 1e-8)) + sampleMean

            if plot:
                Utils.plot(
                    [query.tolist() for query in queriedVectors[0]],
                    "rag.png",
                    ":",
                    self.__contextLength,
                )
                Utils.plot(
                    [queried.tolist()],
                    "ragAvg.png",
                    ":",
                    self.__contextLength,
                )
                Utils.plot(
                    [newSample + predictionNormed.tolist()],
                    "inputAndRag.png",
                    ":",
                    self.__contextLength,
                    rag=True,
                )
                Utils.plot(
                    [sample["value"].tolist() + prediction.tolist()],
                    "pred.png",
                    "-",
                    self.__contextLength,
                )
            return prediction
        else:
            sampleGluonts : ListDataset = ListDataset(
                [{
                    "start": TimeManager.convertTimeFormat(sample["datetime"].iloc[0], timestampFormat, self.__timestampFormat),
                    "target": sample["value"].tolist(),
                }],
                freq=self.__getFrequency(sample["datetime"].iloc[0:2], timestampFormat)
            )
            return next(iter(self.__predictor.predict(sampleGluonts))).quantile(0.5)

    def ragOnlyInference(self, sample : pd.core.frame.DataFrame, dataset : str, softMax : bool = False, cosine : bool = True) -> SampleForecast:
        """
        Method to predict one sample, first columns must be the timestamp and second is the timeseries using rag only
        """
        if len(sample.columns) != 2:
            raise ModelException("MoiraiMoE predictor accepts only two columns, timestamp and timeseries itself")

        timestampFormat : str = self.__datasetsConfig[dataset]["timeformat"]
        sample.columns = ["datetime", "value"]
        queriedVectors : tuple = self.queryVector(sample["value"], k=self.__k, metadata={"dataset" : dataset})
        queried : np.ndarray = self.mergeQueries(queriedVectors) if not softMax else self.mergeQueriesSoftMax(queriedVectors, cosine)
        if queried is not None:
            return queried[self.__contextLength:]
        else:
            sampleGluonts : ListDataset = ListDataset(
                [{
                    "start": TimeManager.convertTimeFormat(sample["datetime"].iloc[0], timestampFormat, self.__timestampFormat),
                    "target": sample["value"].tolist(),
                }],
                freq=self.__getFrequency(sample["datetime"].iloc[0:2], timestampFormat)
            )
            return next(iter(self.__predictor.predict(sampleGluonts))).quantile(0.5)

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
