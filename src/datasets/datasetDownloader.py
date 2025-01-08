from git import Repo
from utils.fileSystem import FileSystem
from exceptions.datasetException import DatasetException

class DatasetDownloader(FileSystem):
    """
    Class to handle datasets downloading
    """
    def __init__(self, datasetName : str, datasetParams : dict):
        super().__init__()
        self.__url : str = datasetParams["url"]
        self.__typeDataset : str = datasetParams["type"]
        self.__datasetPath : str = self._createPath(
            self._getPaths()["datasets"],
            datasetName,
        )

    def __prepareGitRepository(self, url : str):
        """
        Method to prepare git repository
        """
        self._deleteFolder(self.__datasetPath)

        try:
            Repo.clone_from(url, self.__datasetPath)
        except Exception as e:
            raise DatasetException(
                f"Repository {url} could not be cloned, Error: {str(e)}"
            )

    def downloadDataset(self) -> str:
        """
        Method to download dataset

        return str path where dataset is downloaded
        """
        if self.__typeDataset == "gitRepository":
            self.__prepareGitRepository(self.__url)
        else:
            raise DatasetException(
                f"Type {self.__typeDataset} on dataset {datasetName} is not supported"
            )

        return self.__datasetPath
