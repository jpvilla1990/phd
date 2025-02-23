import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from chat_time.model.model import ChatTime
from utils.fileSystem import FileSystem
from exceptions.modelException import ModelException

class ChatTimeModel(FileSystem):
    """
    Class to handle chat time model
    """
    def __init__(
            self,
            contextLenght : int = 32,
            predictionlenght : int = 16,
            modelPath : str = "ChengsenWang/ChatTime-1-7B-Chat",
        ):
        super().__init__()
        self.__model : ChatTime = ChatTime(hist_len=contextLenght, pred_len=predictionlenght, model_path=modelPath)

        self.__contextLenght : int = contextLenght
        self.__predictionlenght : int = predictionlenght

    def inference(self, sample : pd.core.frame.DataFrame) -> np.ndarray:
        """
        Method to predict one sample, first columns must be the timestamp and second is the timeseries
        """
        if len(sample.columns) != 2:
            raise ModelException("ChatTime predictor accepts only two columns, timestamp and timeseries itself")
        
        sample.columns = ["datetime", "value"]

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

        histX : np.ndarray = np.linspace(0, self.__contextLenght-1, self.__contextLenght)
        predX : np.ndarray = np.linspace(self.__contextLenght, self.__contextLenght+self.__predictionlenght-1, self.__predictionlenght)

        plt.figure(figsize=(8, 2), dpi=500)
        plt.plot(histX, sample["value"].values, color='#000000')
        plt.plot(predX, groundTruth["value"].values, color='#000000', label='true')
        plt.plot(predX, prediction, color='#FF7F0E', label='pred')
        plt.axvline(self.__contextLenght, color='red')
        plt.legend(loc="upper left")
        plt.show()
