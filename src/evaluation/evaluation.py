import pandas as pd
import numpy as np
from gluonts.model.forecast import SampleForecast
from datasets.datasets import Datasets
from model.moiraiMoe import MoiraiMoE
from model.chatTime import ChatTime
from utils.fileSystem import FileSystem
from utils.utils import Utils
from datasets.datasetIterator import DatasetIterator

class Evaluation(FileSystem):
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

        meanAbsoluteDeviation : float = np.mean(
            abs(context[:-1] - context[1:])
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
    
    def compileReports(self):
        """
        Method to compile results in a human readable report
        """
        reports : dict = self.__loadReports()

        tables : dict = {}

        for dataset in reports:
            for scenario in reports[dataset]:
                if scenario not in tables:
                    tables.update({
                        scenario : {
                            "scenario" : {},
                            "indices" : [],
                        },
                    })
                if dataset not in tables[scenario]["indices"]:
                    tables[scenario]["indices"].append(dataset)

                totalIterations : int = 0
                for subdataset in reports[dataset][scenario]:
                    for metric in reports[dataset][scenario][subdataset]:
                        if metric == "numberIterations":
                            continue
                        if metric not in tables[scenario]["scenario"]:
                            tables[scenario]["scenario"].update({
                                metric : [],
                            })
                        if len(tables[scenario]["scenario"][metric]) < len(tables[scenario]["indices"]):
                            tables[scenario]["scenario"][metric].append(
                                reports[dataset][scenario][subdataset][metric]["mean"],
                            )
                        else:
                            tables[scenario]["scenario"][metric][-1] += reports[dataset][scenario][subdataset][metric]["mean"] * reports[dataset][scenario][subdataset]["numberIterations"]
        
        print(tables)
        """
        ET:
  128,16:
    ETTh1:
      MAE:
        mean: 2.713830714750401
        median: 1.2220036685466766
      MASE:
        mean: 1.7466749257610454
        median: 1.6631743962960615
      MSE:
        mean: 29.01229773812218
        median: 2.0925267172823894
      normalizedMAE:
        mean: 0.21322683745292131
        median: 0.09601334975827358
      normalizedMSE:
        mean: 2.279508615006335
        median: 0.16441071721487388
      numberIterations: 120939
      """

    def evaluateMoiraiMoE(
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
    
    def evaluateChatTimes(
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
        model : ChatTime = ChatTime(
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