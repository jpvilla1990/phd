import random
import pandas as pd
import numpy as np
from gluonts.dataset.pandas import PandasDataset
from exceptions.datasetException import DatasetException

class DatasetIterator(object):
    """
    Class to handle iterators on the datasets
    """
    def __init__(self, name : str, datasets : dict, datasetConfig : dict):
        self.__datasets : dict = datasets
        self.__datasetConfig : dict = datasetConfig
        self.__name : str = name
        self.__sampleSize : int = 1

        self.__features : dict = {}
        self.__datasetSizes : dict = {}

        self.__indexIterator : dict = {subDataset : [] for subDataset in self.__datasets}

    def __str__(self) -> str:
        return self.__name

    def __getDatasetMetada(self, subdataset : str) -> dict:
        """
        Method to get available features
        """
        datasetMetadata : dict = {
            "numberFeatures" : 0,
            "numberObservations" : 0,
            "mean" : 0.0,
            "std" : 0.0,
        }

        if subdataset not in self.__datasets:
            raise DatasetException(
                f"Subdataset {subdataset} does not exists in dataset {self}"
            )
        
        features : list = pd.read_csv(
            self.__datasets[subdataset],
            nrows=0,
            sep=self.__datasetConfig["separator"],
            decimal=self.__datasetConfig["decimal"],
        ).columns.tolist()

        datasetMetadata["numberFeatures"] = len(features)
        for chunk in pd.read_csv(
            self.__datasets[subdataset],
            sep=self.__datasetConfig["separator"],
            decimal=self.__datasetConfig["decimal"],
            chunksize=10000,
        ):
            datasetMetadata["numberObservations"] += len(chunk)
            datasetMetadata["mean"] += chunk[features[1:]].values.mean() * len(chunk)
            datasetMetadata["std"] += chunk[features[1:]].values.std() * len(chunk)

        datasetMetadata["mean"] /= datasetMetadata["numberObservations"]
        datasetMetadata["std"] /= datasetMetadata["numberObservations"]

        return datasetMetadata

    def setSampleSize(self, sampleSize : int):
        """
        Method to set sample size
        """
        self.__sampleSize = sampleSize

    def getDatasetMetadata(self) -> dict:
        """
        Method to get sizes of all datasets
        """
        self.__datasetMetadata = {subDataset : self.__getDatasetMetada(subDataset) for subDataset in self.__datasets}
        self.__datasetMetadata["mean"] = 0.0
        self.__datasetMetadata["std"] = 0.0

        totalSamples : int = 0
        for subDataset in self.__datasets:
            samples : int = self.__datasetMetadata[subDataset]["numberFeatures"] * self.__datasetMetadata[subDataset]["numberObservations"]
            self.__datasetMetadata["mean"] += self.__datasetMetadata[subDataset]["mean"] * samples
            self.__datasetMetadata["std"] += self.__datasetMetadata[subDataset]["std"] * samples
            totalSamples += samples

        self.__datasetMetadata["mean"] /= totalSamples
        self.__datasetMetadata["std"] /= totalSamples

        return self.__datasetMetadata

    def getAvailableFeatures(self, subdataset : str) -> dict:
        """
        Method to get available features
        """
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

    def loadSample(
            self,
            subdataset : str,
            sampleIndex : int = 0,
            sampleSize : int = 1,
            features : list = [],
        ) -> pd.core.frame.DataFrame:
        """
        Method to load samples
        """
        self.__features[subdataset] = self.getAvailableFeatures(subdataset)

        self.__datasetSizes[subdataset] = self.__getDatasetMetada(subdataset)
        sample : pd.core.frame.DataFrame = None
        featuresIndices : list = None
        maxNumberSamples : int = self.__datasetSizes[subdataset]["numberObservations"]

        if len(features) == 0:
            featuresIndices = [self.__features[subdataset][key] for key in self.__features[subdataset]]
        else:
            featuresIndices = [self.__features[subdataset][feature] for feature in features]

        if sampleIndex <= 0:
            sampleIndex : int = random.randint(1, maxNumberSamples - sampleSize + 1)
        elif sampleIndex > maxNumberSamples - sampleSize + 1:
            sampleSize = maxNumberSamples - sampleIndex + 1

        if subdataset not in self.__datasets:
            raise DatasetException(
                f"Subdataset {subdataset} does not exists in dataset {self}"
            )
        if sampleSize < 1:
            raise DatasetException(
                f"The sampleSize can not be less than 1"
            )

        if sampleSize >= maxNumberSamples:
            sample = pd.read_csv(
                self.__datasets[subdataset],
                sep=self.__datasetConfig["separator"],
                decimal=self.__datasetConfig["decimal"],
                usecols=featuresIndices,
            )

        else:
            sampleIndices : list = range(sampleIndex, sampleIndex + sampleSize)
            sample = pd.read_csv(
                self.__datasets[subdataset],
                sep=self.__datasetConfig["separator"],
                header=None,
                usecols=featuresIndices,
                decimal=self.__datasetConfig["decimal"],
                skiprows=sampleIndex,
                nrows=sampleSize,
            )
            sample.index = sampleIndices

        return sample

    def resetIteration(self, subdataset : str, randomOrder : bool = False):
        """
        Method to reset dataset iteration
        """
        self.__datasetSizes[subdataset] = self.__getDatasetMetada(subdataset)
        self.__indexIterator[subdataset] = []
        maxNumberSamples : int = self.__datasetSizes[subdataset]["numberObservations"]

        indexIterator : list = [index for index in range(1, maxNumberSamples - self.__sampleSize + 2)]

        if randomOrder:
            random.shuffle(indexIterator)
        else:
            indexIterator.sort(reverse=True)

        self.__indexIterator[subdataset] = indexIterator

    def iterateDataset(
            self,
            subdataset : str,
            features : list = [],
        ) -> pd.core.frame.DataFrame:
        """
        Method to iterate through out the whole dataset
        """
        if len(self.__indexIterator[subdataset]) == 0:
            return None

        sample : pd.core.frame.Dataframe = self.loadSample(
            subdataset=subdataset,
            sampleIndex=self.__indexIterator[subdataset][-1],
            sampleSize=self.__sampleSize,
            features=features,
        )

        self.__indexIterator[subdataset].pop()

        return sample

