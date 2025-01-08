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

    def loadSamples(self, subdataset : str, numberSamples : int) -> any:
        """
        Method to load samples
        """
        if subdataset not in self.__datasets:
            raise DatasetException(
                f"Subdataset {subdataset} does not exists in dataset {self}"
            )

        return self.__datasets[subdataset]