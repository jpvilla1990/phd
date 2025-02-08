import pandas as pd
import numpy as np
from gluonts.model.forecast import SampleForecast
from datasets.datasets import Datasets
from model.moiraiMoe import MoiraiMoE
from utils.fileSystem import FileSystem
from utils.utils import Utils
from datasets.datasetIterator import DatasetIterator

class EvaluationMoiraiMoE(FileSystem):
    """
    Class to evaluate models
    """
    def __init__(self):
        super().__init__()

        self.__dataset : Datasets = Datasets()
        self.__datasetMetadata : dict = {}

    def __loadReports(self) -> dict:
        """
        Method to load dataset config
        """
        reports : dict = Utils.readYaml(
            self._getFiles()["evaluationReports"]
        )
        return reports if type(reports) == dict else dict()

    def __writeReports(self, entry : dict):
        """
        Method to write in dataset config
        """
        Utils.writeYaml(
            self._getFiles()["evaluationReports"],
            self.__loadReports() | entry,
        )

    def __getMASE(self, context : np.ndarray, groundTruth : np.ndarray, prediction : np.ndarray) -> float:
        """
        Method to calculate MEAN ABSOLUTE SCALED ERROR
        """
        meanAbsoluteError : float = np.mean(
            abs(prediction - groundTruth),
        )

        mean : float = np.mean(np.append(context,groundTruth))
        meanAbsoluteDeviation : float = np.mean(
            abs(prediction - mean)
        )

        if meanAbsoluteDeviation == 0:
            return None

        return meanAbsoluteError / meanAbsoluteDeviation
    
    def __getNMAE(self, groundTruth : np.ndarray, prediction : np.ndarray) -> float:
        """
        Method to calculate NORMALIZED MEAN ABSOLUTE ERROR
        """
        normalizedMeanAbsoluteError : float = np.mean(
            abs((prediction - groundTruth) / self.__datasetMetadata["std"]),
        )

        return normalizedMeanAbsoluteError

    def evaluate(
            self,
            contextLenght : int,
            predictionLength : int,
            numberSamples : int,
            dataset : str,
            subdataset : str = "",
        ) -> dict:
        """
        Method to evaluate model
        """
        print(f"Evaluating Dataset {dataset}")
        subdatasets : list = []
        model : MoiraiMoE = MoiraiMoE(
            predictionLength = predictionLength,
            contextLenght = contextLenght,
            numSamples = numberSamples,
        )
        iterator : DatasetIterator = self.__dataset.loadDataset(dataset)
        self.__datasetMetadata = iterator.getDatasetMetadata()
        iterator.setSampleSize(contextLenght + predictionLength)

        if subdataset == "":
            datasetConfig : dict = Utils.readYaml(
                self._getFiles()["datasets"]
            )
            subdatasets = list(datasetConfig[dataset].keys())
        else:
            subdatasets.append(subdataset)

        for element in subdatasets:
            print(f"Subdataset {element}")
            reportNMAE : np.ndarray = np.array([])
            reportMASE : np.ndarray = np.array([])
            iterator.resetIteration(element, True)
            features : list = list(iterator.getAvailableFeatures(element).keys())

            while True:
                sample : pd.core.frame.DataFrame = iterator.iterateDataset(element, features)
                if sample is None:
                    break

                for index in range(1,len(features)):
                    pred : SampleForecast = model.inference(sample[[0, index]], dataset)

                    mase : float = self.__getMASE(
                        sample[index].iloc[:contextLenght].values,
                        sample[index].iloc[contextLenght:contextLenght+predictionLength].values,
                        pred.quantile(0.5),
                    )

                    nmae : float = self.__getNMAE(
                        sample[index].iloc[contextLenght:contextLenght+predictionLength].values,
                        pred.quantile(0.5),
                    )

                    if mase:
                        reportMASE = np.append(reportMASE, [mase])
                    if nmae:
                        reportNMAE = np.append(reportNMAE, [nmae])

            reports : dict = self.__loadReports()
            reports.update({
                dataset : {
                    f"{contextLenght},{predictionLength}" : {
                        element : {
                            "MASE" : {
                                "mean" : float(reportMASE.mean()),
                                "median" : float(np.median(reportMASE)),
                            },
                            "normalizedMAE" : {
                                "mean" : float(reportNMAE.mean()),
                                "median" : float(np.median(reportNMAE)),
                            },
                        },
                    },
                },
            })

            self.__writeReports(reports)

        return reports