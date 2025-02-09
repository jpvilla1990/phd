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
    
    def __getMAE(self, groundTruth : np.ndarray, prediction : np.ndarray) -> float:
        """
        Method to calculate MEAN ABSOLUTE ERROR
        """
        meanAbsoluteError : float = np.mean(
            abs((prediction - groundTruth)),
        )

        return meanAbsoluteError
    
    def __getMSE(self, groundTruth : np.ndarray, prediction : np.ndarray) -> float:
        """
        Method to calculate MEAN SQUARED ERROR
        """
        meanAbsoluteError : float = np.mean(
            (prediction - groundTruth) ** 2,
        )

        return meanAbsoluteError

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
            reportMAE : np.ndarray = np.array([])
            reportNMAE : np.ndarray = np.array([])
            reportMSE : np.ndarray = np.array([])
            reportNMSE : np.ndarray = np.array([])
            reportMASE : np.ndarray = np.array([])
            iterations : int = 0
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

                    mae : float = self.__getMAE(
                        sample[index].iloc[contextLenght:contextLenght+predictionLength].values,
                        pred.quantile(0.5),
                    )

                    mse : float = self.__getMSE(
                        sample[index].iloc[contextLenght:contextLenght+predictionLength].values,
                        pred.quantile(0.5),
                    )

                    if mase:
                        reportMASE = np.append(reportMASE, [mase])
                    if mae:
                        reportMAE = np.append(reportMAE, [mae])
                        reportNMAE = np.append(reportNMAE, [abs(mae / self.__datasetMetadata["std"])])
                    if mse:
                        reportMSE = np.append(reportMSE, [mse])
                        reportNMSE = np.append(reportNMSE, [abs(mse / self.__datasetMetadata["std"])])

                    iterations += 1

            reports : dict = self.__loadReports()

            if dataset not in reports:
                reports[dataset] = dict()
            if f"{contextLenght},{predictionLength}" not in reports[dataset]:
                reports[dataset][f"{contextLenght},{predictionLength}"] = dict()

            reports[dataset][f"{contextLenght},{predictionLength}"][element] = {
                "MASE" : {
                    "mean" : float(reportMASE.mean()),
                    "median" : float(np.median(reportMASE)),
                },
                "MAE" : {
                    "mean" : float(reportMAE.mean()),
                    "median" : float(np.median(reportMAE)),
                },
                "normalizedMAE" : {
                    "mean" : float(reportNMAE.mean()),
                    "median" : float(np.median(reportNMAE)),
                },
                "MSE" : {
                    "mean" : float(reportMSE.mean()),
                    "median" : float(np.median(reportMSE)),
                },
                "normalizedMSE" : {
                    "mean" : float(reportNMSE.mean()),
                    "median" : float(np.median(reportNMSE)),
                },
                "numberIterations" : iterations,
            }

            self.__writeReports(reports)

        return reports