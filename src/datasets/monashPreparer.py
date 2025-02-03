import pandas as pd
from utils.fileSystem import FileSystem
from utils.timeManager import TimeManager

class MonashPreparer(FileSystem):
    """
    Class to Handle Monash Dataset Preparation to csv
    """
    def __init__(self, dataset : str):
        super().__init__()
        self.__dataset : dict = self._getConfig()["datasets"][dataset]

    def prepare(self, datasetConfig : dict) -> dict:
        """
        Method to prepare csv with monash dataset
        """
        newDatasetConfig : dict = {}
        separator : str = self.__dataset["separator"]
        decimal : str = self.__dataset["decimal"]
        period : int = TimeManager.getPeriodSeconds(self.__dataset["periodicity"])

        for subdataset in datasetConfig:
            datasetLineFound = False
            with open(datasetConfig[subdataset], "r", encoding="utf-8", errors="replace") as datasetFile:
                while True:
                    nextLine : str = datasetFile.readline()
                    if nextLine is None:
                        break
                    elif nextLine[0:3] == "T1:":
                        datasetLineFound = True

                    if datasetLineFound:
                        if nextLine == "":
                            break
                        feature : str = nextLine.split(":")[0]
                        data : list = nextLine.split(f"{feature}:")[1].split(separator)
                        timestamp : str = data[0].split(":")[0]
                        samples : list = data[1:]

                        subdatasetPath : str = datasetConfig[subdataset].split(".")[0] + f"_{feature}.csv"

                        df : pd.core.frame.DataFrame = pd.DataFrame(columns=["timestamp", feature])

                        for index in range(len(samples)):
                            df.loc[index] = [timestamp, float(samples[index].replace(separator, "."))]

                            timestamp = TimeManager.nextTimeStamp(
                                timestamp,
                                self._getConfig()["monash"]["timeFormat"],
                                period,
                            )

                        df.to_csv(subdatasetPath, sep=separator, decimal=decimal, index=False)
                        newDatasetConfig.update({
                            feature : subdatasetPath,
                        })

        return newDatasetConfig




