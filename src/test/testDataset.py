import pytest
import pandas as pd
from utils.utils import Utils
from utils.fileSystem import FileSystem
from datasets.datasets import Datasets
from datasets.datasetIterator import DatasetIterator

class TestDataset(object):
    """
    Class to handle dataset testcases
    """
    @pytest.fixture(autouse=True)
    def setup(self):
        datasetsUnderTest : list = [
            "ET",
            "electricityUCI",
            "solarEnergy",
        ]
        self.__fileSystem : FileSystem = FileSystem()
        datasetConfig : dict = Utils.readYaml(
            self.__fileSystem._getFiles()["datasets"]
        )
        self.__datasets : dict = {dataset : [] for dataset in datasetsUnderTest}

        datasets : Datasets = Datasets()

        for element in datasetsUnderTest:
            datasets.loadDataset(element)
            subdatasets : list = list(datasetConfig[element].keys())
            self.__datasets[element] = subdatasets

    @pytest.mark.skip(reason="Consumes a lot of bandwith, disabled to test other functionalities.")
    def testDatasetDownload(self, setup):
        """
        Test download dataset works
        """
        dataset : Datasets = Datasets()

        for element in self.__datasets:
            try:
                dataset.loadDataset(element, True)
                assert True
            except:
                assert False

    def testIteratorLoad(self, setup):
        """
        Test iterator is loaded for each dataset
        """
        dataset : Datasets = Datasets()

        for element in self.__datasets:
            iterator : DatasetIterator = dataset.loadDataset(element)
            assert type(iterator) == DatasetIterator
            assert iterator is not None

    def testGetDatasetSize(self, setup):
        """
        Test function to get dataset size works
        """
        dataset : Datasets = Datasets()

        for element in self.__datasets:
            iterator : DatasetIterator = dataset.loadDataset(element)
            sizeDict : dict = iterator.getDatasetSizes()
            for subdataset in self.__datasets[element]:
                assert type(sizeDict[subdataset]) == dict
                assert type(sizeDict[subdataset]["numberFeatures"]) == int
                assert type(sizeDict[subdataset]["numberObservations"]) == int
                assert sizeDict[subdataset]["numberFeatures"] > 0
                assert sizeDict[subdataset]["numberObservations"] > 0

    def testGetFeatures(self, setup):
        """
        Test function get features
        """
        dataset : Datasets = Datasets()

        for element in self.__datasets:
            iterator : DatasetIterator = dataset.loadDataset(element)
            for subdataset in self.__datasets[element]:
                features : dict = iterator.getAvailableFeatures(subdataset)
                assert type(features) == dict
                assert len(features) > 1

    def testGetSample(self, setup):
        """
        Test function to get sample works
        """
        dataset : Datasets = Datasets()

        for element in self.__datasets:
            iterator : DatasetIterator = dataset.loadDataset(element)
            for subdataset in self.__datasets[element]:
                features : list = iterator.getAvailableFeatures(subdataset)
                sample : pd.core.frame.DataFrame = iterator.loadSample(subdataset, 1, 1, features)

                assert type(sample) == pd.core.frame.DataFrame
                assert len(sample.columns) == len(features)
                assert len(sample) == 1
                assert sample.index[0] == 1

    def testIterateDataset(self, setup):
        """
        Test function iterate dataset
        """
        dataset : Datasets = Datasets()

        for element in self.__datasets:
            iterator : DatasetIterator = dataset.loadDataset(element)
            for subdataset in self.__datasets[element]:
                features : list = iterator.getAvailableFeatures(subdataset)
                iterator.setSampleSize(1)
                iterator.resetIteration(subdataset)
                sample : pd.core.frame.Dataframe = iterator.iterateDataset(subdataset, features)

                assert type(sample) == pd.core.frame.DataFrame
                assert len(sample.columns) == len(features)
                assert len(sample) == 1
                assert sample.index[0] == 1