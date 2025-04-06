import pytest
import pandas as pd
import numpy as np
from gluonts.model.forecast import SampleForecast
from utils.utils import Utils
from utils.fileSystem import FileSystem
from datasetsModule.datasets import Datasets
from datasetsModule.datasetIterator import DatasetIterator
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
        self.__timeformats : dict = {
            "ET" : "%Y-%m-%d %H:%M:%S",
            "electricityUCI" : "%Y-%m-%d %H:%M:%S",
            "power" : "%d.%m.%Y %H:%M",
            "m4-monthly" : "%Y-%m-%d %H-%M-%S",
            "solarEnergy" : "%Y-%m-%d %H-%M-%S",
        }
        self.__fileSystem : FileSystem = FileSystem()
        datasetConfig : dict = Utils.readYaml(
            self.__fileSystem._getFiles()["datasets"]
        )
        self.__datasets : dict = {dataset : [] for dataset in datasetsUnderTest}

        datasets : Datasets = Datasets()

        for element in datasetsUnderTest:
            datasetsModule.loadDataset(element)
            subdatasets : list = list(datasetConfig[element].keys())
            self.__datasets[element] = subdatasets

    def testMoiraiMoEInference(self, setup):
        """
        Test function to get sample works
        """
        CONTEXT : int = 32
        PREDICTION_LENGHT : int = 16
        NUMBER_SAMPLES : int = 50
        dataset : Datasets = Datasets()

        for element in self.__datasets:
            iterator : DatasetIterator = dataset.loadDataset(element, True)
            for subdataset in self.__datasets[element]:
                features : dict = iterator.getAvailableFeatures(subdataset)
                sample : pd.core.frame.DataFrame = iterator.loadSample(subdataset, 1, CONTEXT, list(features.keys())[0:2])

                model : MoiraiMoE = MoiraiMoE(
                    predictionLength = PREDICTION_LENGHT,
                    contextLenght = CONTEXT,
                    numSamples = NUMBER_SAMPLES,
                )
                prediction : pd.ndarray = model.inference(
                    sample,
                    self.__timeformats[element],
                ).samples

                assert type(prediction) == np.ndarray
                assert prediction.shape[0] == NUMBER_SAMPLES
                assert prediction.shape[1] == PREDICTION_LENGHT