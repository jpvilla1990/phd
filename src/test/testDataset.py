import time
import pytest
from utils.fileSystem import FileSystem
from datasets.datasets import Datasets

class TestDataset(object):
    """
    Class to handle dataset testcases
    """
    @pytest.fixture(autouse=True)
    def setup(self):
        self.__fileSystem : FileSystem = FileSystem()
        self.__datasets : dict = {
            "ET": [
                "ETTh1",
                "ETTh2",
                "ETTm1",
                "ETTm2",
            ],
            "electricityUCI": [
                "electricity",
            ],
        }

    #@pytest.mark.skip(reason="Consumes a lot of bandwith, disabled to test other functionalities.")
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