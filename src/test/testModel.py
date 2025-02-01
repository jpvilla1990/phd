import pytest
import pandas as pd
import numpy as np
from gluonts.model.forecast import SampleForecast
from utils.utils import Utils
from utils.fileSystem import FileSystem
from datasets.datasets import Datasets
from datasets.datasetIterator import DatasetIterator
from model.moiraiMoe import MoiraiMoE

class TestModel(object):
    """
    Class to handle model testcases
    """
    @pytest.fixture(autouse=True)
    def setup(self):
        datasetsUnderTest : list = [
            "ET",
            "electricityUCI",
            "solarEnergy",
            "m4-monthly",
            "power",
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

    def testMoiraiMoEInference(self, setup):
        """
        Test function to get sample works
        """
        CONTEXT : int = 200
        PREDICTION_LENGHT : int = 20
        NUMBER_SAMPLES : int = 50
        dataset : Datasets = Datasets()

        model : MoiraiMoE = MoiraiMoE(
            predictionLength = PREDICTION_LENGHT,
            contextLenght = CONTEXT,
            numSamples = NUMBER_SAMPLES,
        )
        iterator : DatasetIterator = dataset.loadDataset("power")
        sample : pd.core.frame.DataFrame = iterator.loadSample("power", 1, CONTEXT, ["Date_Time", "Natural Gas"])

        prediction : pd.ndarray = model.inference(
            sample,
        ).samples

        assert type(prediction) == np.ndarray
        assert prediction.shape[0] == NUMBER_SAMPLES
        assert prediction.shape[1] == PREDICTION_LENGHT