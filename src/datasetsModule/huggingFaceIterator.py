import pandas as pd
import random
from datasets import load_dataset

class HuggingFaceIterator(object):
    """
    Class to handle iterators on the datasets
    """
    def __init__(self, name : str, datasets : dict, datasetPath : str, datasetConfig : dict, seed : int = 42):
        random.seed(seed)
        self.__datasets : dict = datasets
        self.__datasetConfig : dict = datasetConfig
        self.__name : str = name
        self.__datasetPath : str = datasetPath

        self.__features : dict = {}
        self.__datasetSizes : dict = {}

    def setSampleSize(self, sampleSize : int):
        """
        Method to set sample size
        """
        self.__sampleSize = sampleSize

    def getAvailableFeatures(self, subdataset : str) -> dict:
        """
        Method to get available features
        """
        dataset = load_dataset(self.__datasetConfig["datasetName"], subdataset, split="train", cache_dir=self.__datasetPath)
        feature_names = list(dataset.features.keys())
        print(dataset[0][feature_names[0]])
        print(dataset[1][feature_names[0]])
        print(dataset[0][feature_names[1]])
        print(dataset[1][feature_names[1]])
        print(dataset[0][feature_names[2]])
        print(dataset[1][feature_names[2]])
        print(dataset[0][feature_names[3]])
        print("#############################################")
        print(dataset[0][feature_names[4]])
        print(feature_names)
        print(len(dataset[0][feature_names[3]]))
        print(len(dataset[0][feature_names[4]]))
        features : dict = {}

        if subdataset not in self.__datasets:
            raise DatasetException(
                f"Subdataset {subdataset} does not exists in dataset {self}"
            )

        features = {value : index for index, value in enumerate(pd.read_csv(
            self.__datasets[subdataset],
            nrows=0,
            sep=self.__datasetConfig["separator"],
            decimal=self.__datasetConfig["decimal"],
        ).columns)}

        return features

    def iterateDataset(
            self,
            train : bool = True,
        ) -> pd.core.frame.DataFrame:
        """
        Method to iterate through out the whole dataset
        """
        category : str = "test"
        if train:
            category = "train"

        if len(self.__indexIterator[subdataset][category]) == 0:
            if "frame" in self.__indexIterator[subdataset]:
                del self.__indexIterator[subdataset]["frame"]
            return None

        sample : pd.core.frame.Dataframe = self.loadSample(
            subdataset=subdataset,
            sampleIndex=self.__indexIterator[subdataset][category][-1],
            sampleSize=self.__sampleSize,
            features=features,
        )

        self.__indexIterator[subdataset][category].pop()

        return sample