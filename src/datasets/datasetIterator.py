import random
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

        self.__datasetSizes : dict = {dataset : self.__getDatasetSize(dataset) for dataset in self.__datasets}

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

        features : list = None

        if subdataset not in self.__datasets:
            raise DatasetException(
                f"Subdataset {subdataset} does not exists in dataset {self}"
            )

        if extension == "csv" or extension == "txt":
            features = pd.read_csv(
                self.__datasets[subdataset],
                nrows=0,
                sep=self.__separator,
                decimal=self.__decimal,
            ).columns.tolist()
        else:
            raise DatasetException(
                f"Extension {extension} not supported, review the subdataset path {self.__datasets[subdataset]}"
            )

        return features

    def loadSamples(
            self,
            subdataset : str,
            sampleSize : int = 1,
            numberSamples : int = 1,
            features : list = []
        ) -> list:
        """
        Method to load samples
        """
        sample : pd.core.frame.DataFrame = None
        samples : list = []
        if subdataset not in self.__datasets:
            raise DatasetException(
                f"Subdataset {subdataset} does not exists in dataset {self}"
            )
        if sampleSize < 1:
            raise DatasetException(
                f"The sampleSize can not be less than 1"
            )
        if len(features) > 0:
            availableFeatures : list = self.getAvailableFeatures(subdataset)
            for feature in features:
                if feature not in availableFeatures:
                    raise DatasetException(
                        f"Feature {feature} does not exists in subdataset {subdataset}"
                    )

        extension : str = self.__getExtension(subdataset)

        for _ in range(numberSamples):
            maxNumberSamples : int = self.__datasetSizes[subdataset]["numberObservations"]
            if sampleSize >= maxNumberSamples:
                if extension == "csv" or extension == "txt":
                    sample = pd.read_csv(
                        self.__datasets[subdataset],
                        sep=self.__separator,
                        decimal=self.__decimal,
                    )

                else:
                    raise DatasetException(
                        f"Extension {extension} not supported, review the subdataset path {self.__datasets[subdataset]}"
                    )
                if len(features) > 0:
                    samples.append(
                        sample[features]
                    )
                else:
                    samples.append(
                        sample
                    )
                    break
            else:
                sampleIndex : int = random.randint(0, maxNumberSamples - sampleSize)
                sampleIndices : list = range(sampleIndex + 1, sampleIndex + sampleSize + 1)
                if extension == "csv" or extension == "txt":
                    sample = pd.read_csv(
                        self.__datasets[subdataset],
                        sep=self.__separator,
                        decimal=self.__decimal,
                        skiprows=lambda x: x not in sampleIndices and x != 0,
                    )
                    sample.index = sampleIndices

                else:
                    raise DatasetException(
                        f"Extension {extension} not supported, review the subdataset path {self.__datasets[subdataset]}"
                    )

                if len(features) > 0:
                    samples.append(
                        sample[features]
                    )
                else:
                    samples.append(
                        sample
                    )

        return samples