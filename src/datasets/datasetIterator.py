import random
import math
import pandas as pd
from exceptions.datasetException import DatasetException

class DatasetIterator(object):
    """
    Class to handle iterators on the datasets
    """
    def __init__(self, name : str, datasets : dict, separator : str, decimal : str):
        self.__datasets : dict = datasets
        self.__name : str = name
        self.__separator : str = separator
        self.__decimal : str = decimal
        self.__sampleSize : int = 1

        self.__indexIterator : dict = {subDataset : [] for subDataset in self.__datasets}

        self.__datasetSizes : dict = {subDataset : self.__getDatasetSize(subDataset) for subDataset in self.__datasets}
        self.__features : dict = {subDataset : self.getAvailableFeatures(subDataset) for subDataset in self.__datasets}

    def __str__(self) -> str:
        return self.__name

    def __getExtension(self, subdataset : str) -> str:
        """
        Method to get extension of subdataset
        """
        return self.__datasets[subdataset].split(".")[-1].lower()
    
    def __getDatasetSize(self, subdataset : str) -> dict:
        """
        Method to get available features
        """
        datasetSize : dict = {
            "numberFeatures" : 0,
            "numberObservations" : 0,
        }
        extension : str = self.__getExtension(subdataset)

        if subdataset not in self.__datasets:
            raise DatasetException(
                f"Subdataset {subdataset} does not exists in dataset {self}"
            )

        if extension == "csv" or extension == "txt":
            datasetSize["numberFeatures"] = len(pd.read_csv(
                self.__datasets[subdataset],
                nrows=0,
                sep=self.__separator,
                decimal=self.__decimal,
            ).columns.tolist())
            for chunk in pd.read_csv(
                self.__datasets[subdataset],
                sep=self.__separator,
                decimal=self.__decimal,
                chunksize=10000,
            ):
                datasetSize["numberObservations"] += len(chunk)

        else:
            raise DatasetException(
                f"Extension {extension} not supported, review the subdataset path {self.__datasets[subdataset]}"
            )

        return datasetSize
    
    def setSampleSize(self, sampleSize : int):
        """
        Method to set sample size
        """
        self.__sampleSize = sampleSize
    
    def getDatasetSizes(self) -> dict:
        """
        Method to get sizes of all datasets
        """
        return self.__datasetSizes

    def getAvailableFeatures(self, subdataset : str) -> list:
        """
        Method to get available features
        """
        extension : str = self.__getExtension(subdataset)

        features : dict = {}

        if subdataset not in self.__datasets:
            raise DatasetException(
                f"Subdataset {subdataset} does not exists in dataset {self}"
            )

        if extension == "csv" or extension == "txt":
            features = {value : index for index, value in enumerate(pd.read_csv(
                self.__datasets[subdataset],
                nrows=0,
                sep=self.__separator,
                decimal=self.__decimal,
            ).columns)}

        else:
            raise DatasetException(
                f"Extension {extension} not supported, review the subdataset path {self.__datasets[subdataset]}"
            )

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
        sample : pd.core.frame.DataFrame = None
        featuresIndices : list = None
        maxNumberSamples : int = self.__datasetSizes[subdataset]["numberObservations"]

        if len(features) == 0:
            featuresIndices = [value for _, value in enumerate(self.__features[subdataset])]
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

        extension : str = self.__getExtension(subdataset)
        if sampleSize >= maxNumberSamples:
            if extension == "csv" or extension == "txt":
                sample = pd.read_csv(
                    self.__datasets[subdataset],
                    sep=self.__separator,
                    decimal=self.__decimal,
                    usecols=featuresIndices,
                )

            else:
                raise DatasetException(
                    f"Extension {extension} not supported, review the subdataset path {self.__datasets[subdataset]}"
                )
        else:
            sampleIndices : list = range(sampleIndex, sampleIndex + sampleSize)
            if extension == "csv" or extension == "txt":
                sample = pd.read_csv(
                    self.__datasets[subdataset],
                    sep=self.__separator,
                    header=None,
                    usecols=featuresIndices,
                    decimal=self.__decimal,
                    skiprows=sampleIndex,
                    nrows=sampleSize,
                )
                sample.index = sampleIndices

            else:
                raise DatasetException(
                    f"Extension {extension} not supported, review the subdataset path {self.__datasets[subdataset]}"
                )

        return sample
    
    def resetIteration(self, subdataset : str, randomOrder : bool = False):
        """
        Method to reset dataset iteration
        """
        self.__indexIterator[subdataset] = []
        maxNumberSamples : int = self.__datasetSizes[subdataset]["numberObservations"]

        indexIterator : list = [index for index in range(1, maxNumberSamples + 1, self.__sampleSize)]

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

