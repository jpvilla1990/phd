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
        plt.axvline(hist_len, color='red')
        plt.legend(loc="upper left")
        plt.show()

dataset = "Traffic"
hist_len = 120
pred_len = 24
model_path = "ChengsenWang/ChatTime-1-7B-Chat"

df = pd.read_csv(f"./ChatTime/dataset/{dataset}.csv")
hist_data = np.array(df["Hist"].apply(eval).values.tolist())[:, -hist_len:][0]
pred_data = np.array(df["Pred"].apply(eval).values.tolist())[:, :pred_len][0]

model = ChatTime(hist_len=hist_len, pred_len=pred_len, model_path=model_path)

out = model.predict(hist_data)

hist_x = np.linspace(0, hist_len-1, hist_len)
pred_x = np.linspace(hist_len, hist_len+pred_len-1, pred_len)

plt.figure(figsize=(8, 2), dpi=500)
plt.plot(hist_x, hist_data, color='#000000')
plt.plot(pred_x, pred_data, color='#000000', label='true')
plt.plot(pred_x, out, color='#FF7F0E', label='pred')
plt.axvline(hist_len, color='red')
plt.legend(loc="upper left")
plt.show()