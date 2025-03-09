import pandas as pd
from utils.utils import Utils
from utils.fileSystem import FileSystem
from datasets.datasets import Datasets
from datasets.datasetIterator import DatasetIterator
from model.moiraiMoe import MoiraiMoE

class VectorDBIngestions(FileSystem):
    """
    Class to handle vector DB ingestion for all datasets
    """
    def __init__(self):
        super().__init__()
        self.__dataset : Datasets = Datasets()

    def __loadDatabaseTracking(self) -> dict:
        """
        Method to load dataset config
        """
        reports : dict = Utils.readYaml(
            self._getFiles()["databaseTracking"]
        )
        return reports if type(reports) == dict else dict()

    def __writeDatabaseTracking(self, entry : dict):
        """
        Method to write in dataset config
        """
        Utils.writeYaml(
            self._getFiles()["databaseTracking"],
            self.__loadDatabaseTracking() | entry,
        )

    def ingestDatasetMoiraiMoE(self, dataset : str, collectionName : str, contextLength : int, predictionLength : int):
        """
        Method to ingest dataset in a collection
        """
        model : MoiraiMoE = MoiraiMoE(
            predictionLength = predictionLength,
            contextLength = contextLength,
            numSamples = 100,
            collectionName = collectionName,
        )
        iterator : DatasetIterator = self.__dataset.loadDataset(dataset)
        iterator.setSampleSize(contextLength + predictionLength)

        datasetConfig : dict = Utils.readYaml(
            self._getFiles()["datasets"]
        )
        subdatasets : list = list(datasetConfig[dataset].keys())
        
        iterations : int = 0

        model.deleteDataset(dataset)

        for element in subdatasets:
            try:
                print(f"Subdataset {element}")
                iterator.resetIteration(element, True)
                features : list = list(iterator.getAvailableFeatures(element).keys())

                while True:
                    sample : pd.core.frame.DataFrame = iterator.iterateDataset(element, features)
                    if sample is None:
                        break
                    if len(sample) < predictionLength + contextLength:
                        break

                    for index in range(1,len(features)):
                        model.ingestVector(
                            sample[index].iloc[:contextLength].values,
                            sample[index].iloc[contextLength:contextLength+predictionLength].values,
                            dataset,
                        )

                        iterations += 1

                if iterations <= 0:
                    continue

            except Exception as e:
                print("Exception: " + str(e))
                continue

        databaseTracking : dict = self.__loadDatabaseTracking()
        if collectionName not in databaseTracking:
            databaseTracking[collectionName] = {}

        databaseTracking[collectionName][dataset] = iterations
        self.__writeDatabaseTracking(databaseTracking)

    def ingestDatasetsMoiraiMoE(self):
        """
        Method to ingest all datasets to MoiraiMoE
        """
        collections : dict = self._getConfig()["vectorDatabase"]["collections"]
        for collection in collections:
            if "moiraiMoE" not in collection:
                continue

            for dataset in collections[collection]["datasets"]:
                databaseTracking : dict = self.__loadDatabaseTracking()
                if collection in databaseTracking:
                    if dataset in databaseTracking[collection][dataset]: # Skip if the dataset is already ingested in collection
                        continue

                self.ingestDatasetMoiraiMoE(
                    dataset,
                    collection,
                    collections[collection]["context"],
                    collections[collection]["prediction"],
                )
