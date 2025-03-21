import pandas as pd
import numpy as np
import concurrent.futures
import random
from gluonts.model.forecast import SampleForecast
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors
from datasets.datasets import Datasets
from model.moiraiMoe import MoiraiMoE
from model.chatTime import ChatTimeModel
from utils.fileSystem import FileSystem
from utils.utils import Utils
from datasets.datasetIterator import DatasetIterator

class Evaluation(FileSystem):
    """
    Class to evaluate models
    """
    def __init__(self):
        super().__init__()
        random.seed(self._getConfig()["seed"])
        self.__dataset : Datasets = Datasets()
        self.__datasetMetadata : dict = {}

    def __loadReport(self, report : str) -> dict:
        """
        Method to load dataset config
        """
        reports : dict = Utils.readYaml(
            self._getFiles()[report]
        )
        return reports if type(reports) == dict else dict()

    def __writeReport(self, entry : dict, report : str):
        """
        Method to write in dataset config
        """
        Utils.writeYaml(
            self._getFiles()[report],
            self.__loadReport(report) | entry,
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
    
    def compileReports(self, reportOriginName : str = "evaluationReportsMoiraiMoE", reportTargetName : str = "evaluationFinalReport"):
        """
        Method to compile results in a human readable report
        """
        report : dict = self.__loadReport(reportOriginName)

        tables : dict = {}

        for dataset in report:
            for scenario in report[dataset]:
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
                for subdataset in report[dataset][scenario]:
                    for metric in report[dataset][scenario][subdataset]:
                        if metric == "numberIterations":
                            continue
                        if metric not in tables[scenario]["scenario"]:
                            tables[scenario]["scenario"].update({
                                metric : [],
                            })
                        if len(tables[scenario]["scenario"][metric]) < len(tables[scenario]["indices"]):
                            tables[scenario]["scenario"][metric].append(
                                report[dataset][scenario][subdataset][metric]["mean"],
                            )
                        else:
                            tables[scenario]["scenario"][metric][-1] = ((tables[scenario]["scenario"][metric][-1] * totalIterations) + (report[dataset][scenario][subdataset][metric]["mean"] * report[dataset][scenario][subdataset]["numberIterations"])) / (totalIterations + report[dataset][scenario][subdataset]["numberIterations"])
                    totalIterations += report[dataset][scenario][subdataset]["numberIterations"]

                if "numberIterations" not in tables[scenario]["scenario"]:
                    tables[scenario]["scenario"].update({
                        "numberIterations" : [],
                    })

                tables[scenario]["scenario"]["numberIterations"].append(totalIterations)

        elements : list = []
        doc : SimpleDocTemplate = SimpleDocTemplate(self._getFiles()[reportTargetName], pagesize=letter)
        for scenario in tables:
            df : pd.core.frame.DataFrame = pd.DataFrame(tables[scenario]["scenario"], index=tables[scenario]["indices"]).round(6)
            elements.append(Table([[f"{reportOriginName} Context Lenght, Prediction Lenght = {scenario}"]], colWidths=[400]))

            tableData = [["Index"] + df.columns.tolist()]
            for index, row in df.iterrows():
                tableData.append([index] + row.tolist())

            table : Table = Table(tableData)

            style = TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ])
            table.setStyle(style)

            elements.append(table)
            elements.append(Table([[""]], colWidths=[400]))  # Add space between tables

        doc.build(elements)

    def evaluateMoiraiMoE(
            self,
            contextLength : int,
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
            contextLength = contextLength,
            numSamples = numberSamples,
        )
        iterator : DatasetIterator = self.__dataset.loadDataset(dataset)
        self.__datasetMetadata = iterator.getDatasetMetadata()
        iterator.setSampleSize(contextLength + predictionLength)

        if subdataset == "":
            datasetConfig : dict = Utils.readYaml(
                self._getFiles()["datasets"]
            )
            subdatasets = list(datasetConfig[dataset].keys())
        else:
            subdatasets.append(subdataset)

        for element in subdatasets:
            try:
                print(f"Subdataset {element}")
                reportMAE : np.ndarray = np.array([])
                reportNMAE : np.ndarray = np.array([])
                reportMSE : np.ndarray = np.array([])
                reportNMSE : np.ndarray = np.array([])
                reportMASE : np.ndarray = np.array([])
                iterations : int = 0
                iterator.resetIteration(element, True, trainPartition=self._getConfig()["trainPartition"])
                features : list = list(iterator.getAvailableFeatures(element).keys())

                while True:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        futureSample : concurrent.futures._base.Future = executor.submit(
                            iterator.iterateDataset,
                            element,
                            features,
                            False,
                        )
                        sample : pd.core.frame.DataFrame = futureSample.result()
                        if sample is None:
                            break
                        if len(sample) < predictionLength + contextLength:
                            break

                        indexes : list = [index for index in range(1,len(features))]
                        random.shuffle(indexes)
                        for i in range(len(indexes)):
                            index : int = indexes[i]
                            if sample[index].isna().any().any():
                                continue
                            pred : SampleForecast = model.inference(sample[[0, index]].iloc[:contextLength], dataset)

                            mase : float = self.__getMASE(
                                sample[index].iloc[:contextLength].values,
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                pred.quantile(0.5),
                            )

                            mae : float = self.__getMAE(
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                pred.quantile(0.5),
                            )

                            mse : float = self.__getMSE(
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                pred.quantile(0.5),
                            )

                            if mase:
                                reportMASE = np.append(reportMASE, [mase])
                            if mae:
                                reportMAE = np.append(reportMAE, [mae])
                                reportNMAE = np.append(reportNMAE, [abs(mae / self.__datasetMetadata["std"])])
                            if mse:
                                reportMSE = np.append(reportMSE, [mse])
                                reportNMSE = np.append(reportNMSE, [abs(mse / (self.__datasetMetadata["std"] ** 2))])

                            iterations += 1

                if iterations <= 0:
                    continue

                report : dict = self.__loadReport("evaluationReportsMoiraiMoE")

                if dataset not in report:
                    report[dataset] = dict()
                if f"{contextLength},{predictionLength}" not in report[dataset]:
                    report[dataset][f"{contextLength},{predictionLength}"] = dict()

                report[dataset][f"{contextLength},{predictionLength}"][element] = {
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

                self.__writeReport(report, "evaluationReportsMoiraiMoE")

            except Exception as e:
                print("Exception: " + str(e))
                continue

        return report
    
    def evaluateMoiraiMoERag(
            self,
            contextLength : int,
            predictionLength : int,
            numberSamples : int,
            dataset : str,
            collection : str,
            subdataset : str = "",
        ) -> dict:
        """
        Method to evaluate model
        """
        print(f"Evaluating Dataset {dataset}")
        subdatasets : list = []
        model : MoiraiMoE = MoiraiMoE(
            predictionLength = predictionLength,
            contextLength = contextLength,
            numSamples = numberSamples,
            rag=True,
            collectionName=collection,
        )
        iterator : DatasetIterator = self.__dataset.loadDataset(dataset)
        self.__datasetMetadata = iterator.getDatasetMetadata()
        iterator.setSampleSize(contextLength + predictionLength)

        if subdataset == "":
            datasetConfig : dict = Utils.readYaml(
                self._getFiles()["datasets"]
            )
            subdatasets = list(datasetConfig[dataset].keys())
        else:
            subdatasets.append(subdataset)

        for element in subdatasets:
            try:
                print(f"Subdataset {element}")
                reportMAE : np.ndarray = np.array([])
                reportNMAE : np.ndarray = np.array([])
                reportMSE : np.ndarray = np.array([])
                reportNMSE : np.ndarray = np.array([])
                reportMASE : np.ndarray = np.array([])
                iterations : int = 0
                iterator.resetIteration(element, True, trainPartition=self._getConfig()["trainPartition"])
                features : list = list(iterator.getAvailableFeatures(element).keys())

                while True:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        futureSample : concurrent.futures._base.Future = executor.submit(
                            iterator.iterateDataset,
                            element,
                            features,
                            False,
                        )
                        sample : pd.core.frame.DataFrame = futureSample.result()
                        if sample is None:
                            break
                        if len(sample) < predictionLength + contextLength:
                            break

                        indexes : list = [index for index in range(1,len(features))]
                        random.shuffle(indexes)
                        for i in range(len(indexes)):
                            index : int = indexes[i]
                            if sample[index].isna().any().any():
                                continue
                            pred : SampleForecast = model.ragInference(sample[[0, index]].iloc[:contextLength], dataset)

                            mase : float = self.__getMASE(
                                sample[index].iloc[:contextLength].values,
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                pred.quantile(0.5),
                            )

                            mae : float = self.__getMAE(
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                pred.quantile(0.5),
                            )

                            mse : float = self.__getMSE(
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                pred.quantile(0.5),
                            )

                            if mase:
                                reportMASE = np.append(reportMASE, [mase])
                            if mae:
                                reportMAE = np.append(reportMAE, [mae])
                                reportNMAE = np.append(reportNMAE, [abs(mae / self.__datasetMetadata["std"])])
                            if mse:
                                reportMSE = np.append(reportMSE, [mse])
                                reportNMSE = np.append(reportNMSE, [abs(mse / (self.__datasetMetadata["std"] ** 2))])

                            iterations += 1

                if iterations <= 0:
                    continue

                report : dict = self.__loadReport("evaluationReportsMoiraiMoERag")

                if dataset not in report:
                    report[dataset] = dict()
                if f"{contextLength},{predictionLength}" not in report[dataset]:
                    report[dataset][f"{contextLength},{predictionLength}"] = dict()

                report[dataset][f"{contextLength},{predictionLength}"][element] = {
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

                self.__writeReport(report, "evaluationReportsMoiraiMoERag")

            except Exception as e:
                print("Exception: " + str(e))
                continue

        return report

    def evaluateChatTimes(
            self,
            contextLength : int,
            predictionLength : int,
            dataset : str,
            subdataset : str = "",
        ) -> dict:
        """
        Method to evaluate model
        """
        print(f"Evaluating Dataset {dataset}")
        subdatasets : list = []
        model : ChatTimeModel = ChatTimeModel(
            predictionLength = predictionLength,
            contextLength = contextLength,
        )
        iterator : DatasetIterator = self.__dataset.loadDataset(dataset)
        self.__datasetMetadata = iterator.getDatasetMetadata()
        iterator.setSampleSize(contextLength + predictionLength)

        if subdataset == "":
            datasetConfig : dict = Utils.readYaml(
                self._getFiles()["datasets"]
            )
            subdatasets = list(datasetConfig[dataset].keys())
        else:
            subdatasets.append(subdataset)

        for element in subdatasets:
            try:
                print(f"Subdataset {element}")
                reportMAE : np.ndarray = np.array([])
                reportNMAE : np.ndarray = np.array([])
                reportMSE : np.ndarray = np.array([])
                reportNMSE : np.ndarray = np.array([])
                reportMASE : np.ndarray = np.array([])
                iterations : int = 0
                iterator.resetIteration(element, True, trainPartition=self._getConfig()["trainPartition"])
                features : list = list(iterator.getAvailableFeatures(element).keys())

                while True:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        futureSample : concurrent.futures._base.Future = executor.submit(
                            iterator.iterateDataset,
                            element,
                            features,
                            False,
                        )
                        sample : pd.core.frame.DataFrame = futureSample.result()
                        if sample is None:
                            break

                        if len(sample) < predictionLength + contextLength:
                            break

                        indexes : list = [index for index in range(1,len(features))]
                        random.shuffle(indexes)
                        for i in range(len(indexes)):
                            index : int = indexes[i]
                            if sample[index].isna().any().any():
                                continue
                            pred : np.ndarray = model.inference(sample[[index]].iloc[:contextLength])

                            mase : float = self.__getMASE(
                                sample[index].iloc[:contextLength].values,
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                pred,
                            )

                            mae : float = self.__getMAE(
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                pred,
                            )

                            mse : float = self.__getMSE(
                                sample[index].iloc[contextLength:contextLength+predictionLength].values,
                                pred,
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

                if iterations <= 0:
                    continue

                report : dict = self.__loadReport("evaluationReportsChatTime")

                if dataset not in report:
                    report[dataset] = dict()
                if f"{contextLength},{predictionLength}" not in report[dataset]:
                    report[dataset][f"{contextLength},{predictionLength}"] = dict()

                report[dataset][f"{contextLength},{predictionLength}"][element] = {
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

                self.__writeReport(report, "evaluationReportsChatTime")

            except Exception as e:
                print("Exception: " + str(e))
                continue

        return report