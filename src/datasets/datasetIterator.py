import pandas as pd
from exceptions.datasetException import DatasetException

class DatasetIterator(object):
    """
    Class to handle iterators on the datasets
    """
    def __init__(self, name : str, datasets : dict):
        self.__datasets : dict = datasets
        self.__name : str = name

    def __str__(self) -> str:
        return self.__name

    def __getExtension(self, subdataset : str) -> str:
        """
        Method to get extension of subdataset
        """
        return self.__datasets[subdataset].split(".")[-1].lower()

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

        if extension == "csv":
            features = pd.read_csv(self.__datasets[subdataset], nrows=0).columns.tolist()

        else:
            raise DatasetException(
                f"Extension {extension} not supported, review the subdataset path {self.__datasets[subdataset]}"
            )

        return features

    def loadSamples(self, subdataset : str, numberSamples : int = 0, features : list = []) -> pd.core.frame.DataFrame:
        """
        Method to load samples
        """
        sample : pd.core.frame.DataFrame = None
        if subdataset not in self.__datasets:
            raise DatasetException(
                f"Subdataset {subdataset} does not exists in dataset {self}"
            )

        extension : str = self.__getExtension(subdataset)

        if extension == "csv":
            sample = pd.read_csv(self.__datasets[subdataset])

        else:
            raise DatasetException(
                f"Extension {extension} not supported, review the subdataset path {self.__datasets[subdataset]}"
            )

        if len(features) > 0:
            availableFeatures : list = self.getAvailableFeatures(subdataset)
            for feature in features:
                if feature not in availableFeatures:
                    raise DatasetException(
                        f"Feature {feature} does not exists in subdataset {subdataset}"
                    )
            sample = sample[features]

        if numberSamples <= 0:
            return sample
        elif numberSamples >= len(sample):
            return sample
        else:
            return sample.sample(n=numberSamples)