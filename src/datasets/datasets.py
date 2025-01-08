import os
from utils.fileSystem import FileSystem
from utils.utils import Utils
from exceptions.datasetException import DatasetException

class Datasets(FileSystem):
    """
    Class to load datasets
    """
    def __init__(self):
        super().__init__()
        self.__datasets : dict = self._getConfig()["datasets"]
        self.__datasetConfig : dict = self.__loadDatasetConfig()

    def __loadDatasetConfig(self) -> dict:
        """
        Method to load dataset config
        """
        return Utils.readYaml(
            self._getFiles()["datasets"]
        )
    
    def __verifyDatasetIsDownloaded(self, dataset : str, type : str) -> bool:
        """
        Method to verify if dataset is already downloaded
        """
        if dataset in self.__datasetConfig:
            return True
        else:
            datasetConfigDict : dict = {
                "path" : os.path.join(self._getPaths["datasets"], dataset)
            }
            self.__datasetConfig[dataset] = datasetConfigDict
            return False

    def loadDataset(self, dataset : str, forceDownload : bool = False):
        """
        Method to load dataset
        """
        self.__datasetConfig : dict = self.__loadDatasetConfig()

        if dataset not in self.__datasets:
            raise DatasetException(f"Dataset {dataset} does not exists")

        datasetDict : dict = self.__datasets[dataset]

        type : str = datasetDict["type"]

        if type == "gitRepository":
            print(datasetDict["url"])
        else:
            raise DatasetException(f"Type {type} on dataset {dataset} is not supported")

        datasetDownloaded : bool = self.__verifyDatasetIsDownloaded(dataset, type)