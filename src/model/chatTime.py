import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline,
)
import torch
import torch.nn as nn

from chat_time.model.model import ChatTime
from utils.fileSystem import FileSystem

from vectorDB.vectorDB import vectorDB
from exceptions.modelException import ModelException

class ChatTimeEmbeddings(nn.Module):
    def __init__(self, chatTime : ChatTime):
        super().__init__()
        self.__device : str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__targetDevice : str = "cpu"
        self.model : LlamaForCausalLM = chatTime.model
        self.tokenizer : LlamaTokenizer = chatTime.tokenizer

        x = self.tokenizer(a)
        y = self.model.model.embed_tokens(torch.Tensor(x["input_ids"]).long().to(self.__device))

    def forward(
            self,
            x : str,
        ) -> torch.Tensor:
        """
        Define the forward pass based on the selected layers.
        """
        x : list = self.tokenizer(x)["input_ids"]
        return self.model.model.embed_tokens(torch.Tensor(x).long().to(self.__device))

    def inference(
            self,
            x : str,
        ) -> torch.Tensor:
        """
        Inference
        """
        output : torch.Tensor = None
        with torch.no_grad():
            output = self(x)

        return output.to(self.__targetDevice)

class ChatTimeModel(FileSystem):
    """
    Class to handle chat time model
    """
    def __init__(
        self,
        contextLength : int = 32,
        predictionLength : int = 16,
        modelPath : str = "ChengsenWang/ChatTime-1-7B-Chat",
    ):
        super().__init__()
        self.__model : ChatTime = ChatTime(hist_len=contextLength, pred_len=predictionLength, model_path=modelPath)

        self.__contextLength : int = contextLength
        self.__predictionlength : int = predictionLength

        self.__chatTimeEmbeddings : ChatTimeEmbeddings = ChatTimeEmbeddings(self.__model)
        self.__vectorDB : vectorDB = vectorDB()

    def setRagCollection(self, collectionName : str, dataset : str):
        """
        Method to set RAG collection
        """
        self.__vectorDB.setCollection(
            collectionName,
            dataset,
            self.__chatTimeEmbeddings.inference,
        )

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

    def inference(self, sample : pd.core.frame.DataFrame) -> np.ndarray:
        """
        Method to predict one sample, first columns must be the timestamp and second is the timeseries
        """
        if len(sample.columns) != 1:
            raise ModelException("ChatTime predictor accepts only one column")

        sample.columns = ["value"]

        return self.__model.predict(sample["value"].values)

    def ragInference(self, sample : pd.core.frame.DataFrame, dataset : str) -> SampleForecast:
        """
        Method to predict one sample, first columns must be the timestamp and second is the timeseries
        """
        if len(sample.columns) != 1:
            raise ModelException("ChatTime predictor accepts only one column")

        sample.columns = ["value"]
        queriedVectors : tuple = self.queryVector(sample["value"], k=self.__k, metadata={"dataset" : dataset})
        queried : np.ndarray = self.mergeQueries(queriedVectors)
        if queried is not None:
            sampleNp : np.ndarray = sample["value"].to_numpy() #
            queriedMean, queriedStd = np.mean(queried), np.std(queried)
            sampleMean, sampleStd = np.mean(sampleNp), np.std(sampleNp)

            queryNormed : np.ndarray = (queried - queriedMean) / (queriedStd + 1e-8)
            queryDenormed : np.ndarray = (queryNormed * sampleStd) + sampleMean ##
            return self.__model.predict(newSample)
        else:
            return self.__model.predict(sample["value"].values)

    def plotSample(self, sample : pd.core.frame.DataFrame, groundTruth : pd.core.frame.DataFrame, dataset : str):
        """
        Method to plot sample, first columns must be the timestamp and second is the timeseries
        """
        if len(sample.columns) != 2 or len(groundTruth.columns) != 2:
            raise ModelException("MoiraiMoE predictor accepts only two columns, timestamp and timeseries itself")

        sample.columns = ["datetime", "value"]
        groundTruth.columns = ["datetime", "value"]

        prediction : np.ndarray = self.__model.predict(sample["value"].values)

        histX : np.ndarray = np.linspace(0, self.__contextLength-1, self.__contextLength)
        predX : np.ndarray = np.linspace(self.__contextLength, self.__contextLength+self.__predictionlength-1, self.__predictionlength)

        plt.figure(figsize=(8, 2), dpi=500)
        plt.plot(histX, sample["value"].values, color='#000000')
        plt.plot(predX, groundTruth["value"].values, color='#000000', label='true')
        plt.plot(predX, prediction, color='#FF7F0E', label='pred')
        plt.axvline(self.__contextLength, color='red')
        plt.legend(loc="upper left")
        plt.show()
