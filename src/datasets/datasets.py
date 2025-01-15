import os
from utils.fileSystem import FileSystem
from utils.utils import Utils
from exceptions.datasetException import DatasetException
from datasets.datasetDownloader import DatasetDownloader
from datasets.datasetIterator import DatasetIterator

class Datasets(FileSystem):
    """
    Class to load datasets
    """
    def __init__(self):
        super().__init__()
        self.__datasets : dict = self._getConfig()["datasets"]

    def __loadDatasetConfig(self) -> dict:
        """
        Method to load dataset config
        """
        datasetConfig : dict = Utils.readYaml(
            self._getFiles()["datasets"]
        )
        return datasetConfig if type(datasetConfig) == dict else dict()

    def __writeDatasetConfig(self, entry : dict):
        """
        Method to write in dataset config
        """
        Utils.writeYaml(
            self._getFiles()["datasets"],
            self.__loadDatasetConfig() | entry,
        )
    
    def __verifyDatasetIsDownloaded(self, dataset : str) -> bool:
        """
        Method to verify if dataset is already downloaded
        """
        if dataset in self.__loadDatasetConfig():
            return True
        else:
            return False

    def loadDataset(self, dataset : str, forceDownload : bool = False):
        """
        Method to load dataset
        """
        if dataset not in self.__datasets:
            raise DatasetException(f"Dataset {dataset} does not exists")

        if not self.__verifyDatasetIsDownloaded(dataset) or forceDownload:
            datasetDownloader : DatasetDownloader = DatasetDownloader(
                dataset,
                self.__datasets[dataset],
            )
            datasetPath : str = datasetDownloader.downloadDataset()

            datasetConfig : dict = {
                subDataset: os.path.join(
                    datasetPath,
                    *self.__datasets[dataset]["subdatasets"][subDataset]
                ) for subDataset in self.__datasets[dataset]["subdatasets"]
            }

            self.__writeDatasetConfig(
                {dataset : datasetConfig}
            )
        
        return DatasetIterator(
            dataset,
            self.__loadDatasetConfig()[dataset],
            self.__datasets[dataset]["separator"],
            self.__datasets[dataset]["decimal"],
        )
