import random
import concurrent.futures
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datasetsModule.datasetIterator import DatasetIterator
from datasetsModule.datasets import Datasets
from utils.fileSystem import FileSystem
from utils.utils import Utils

class AnalyseDataset(FileSystem):
    def __init__(self):
        super().__init__()
        random.seed(self._getConfig()["seed"])
        self.__dataset : Datasets = Datasets()

    def loadSubset(self, dataset : str, contextLength : int, predictionLength : int = 16, trainSet : bool = False) -> dict:
        """
        Method to load a subset of the dataset
        """
        maxTestSamples : int = self._getConfig()["maxTestSamples"]
        subdatasets : list = []
        iterator : DatasetIterator = self.__dataset.loadDataset(dataset)
        iterator.setSampleSize(contextLength + predictionLength)

        datasetConfig : dict = Utils.readYaml(
            self._getFiles()["datasets"]
        )
        subdatasets = list(datasetConfig[dataset].keys())

        samples : list = []

        maxTestSamplesPerSubdataset : int = int(maxTestSamples / len(subdatasets))
        for element in subdatasets:
            try:
                print(f"Subdataset {element}")
                iterations : int = 0
                running : bool = True
                iterator.resetIteration(element, True, trainPartition=self._getConfig()["trainPartition"])
                metadata : dict = iterator.getDatasetMetadata()
                features : list = list(iterator.getAvailableFeatures(element).keys())

                while running:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        futureSample : concurrent.futures._base.Future = executor.submit(
                            iterator.iterateDataset,
                            element,
                            features,
                            trainSet,
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

                            samples.append(sample[index].iloc[:contextLength + predictionLength])

                            iterations += 1

                            if iterations >= maxTestSamplesPerSubdataset:
                                running = False
                                break

                if iterations <= 0:
                    continue

            except Exception as e:
                raise e
                print("Exception: " + str(e))
                continue

        return pd.DataFrame(np.vstack(samples))

    def analyseDataset(self, dataset: str, contextLength: int = 32, predictionLength: int = 16):
        """
        Method to analyse a dataset
        """
        trainSet : pd.DataFrame = self.loadSubset(dataset, contextLength, predictionLength, True)
        testSet : pd.DataFrame = self.loadSubset(dataset, contextLength, predictionLength, False)

        train_stats = trainSet.stack().describe().rename("Train Set")
        test_stats = testSet.stack().describe().rename("Test Set")

        combined_stats = pd.concat([train_stats, test_stats], axis=1)

        combined_stats.to_csv(f"datasetAnalysis/{dataset}_{contextLength}_{predictionLength}_statistics.csv")

        plt.figure(figsize=(10, 6))

        for i in range(min(len(trainSet), len(testSet))):
            plt.plot(trainSet.iloc[i].values, label=f"Dataset 1 - Row {i}", color="red", linestyle="-")
            plt.plot(testSet.iloc[i].values, label=f"Dataset 2 - Row {i}", color="blue", linestyle="--")

        plt.title("Comparison")
        plt.xlabel("Time Index")
        plt.ylabel("Value")

        plt.savefig(f"datasetAnalysis/{dataset}_{contextLength}_{predictionLength}_full_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()

        train_mean = trainSet.mean(axis=0)
        test_mean = testSet.mean(axis=0)

        plt.figure(figsize=(10, 6))
        plt.plot(train_mean.values, color="red", label="Train Mean")
        plt.plot(test_mean.values, color="blue", label="Test Mean")

        plt.savefig(f"datasetAnalysis/{dataset}_{contextLength}_{predictionLength}_mean_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()

if __name__ == "__main__":
    dataset = "ET"
    contextLength = 32
    predictionLength = 16
    element = "value"
    
    analyser = AnalyseDataset()
    analyser.loadSubset(dataset, contextLength, predictionLength, trainSet=False)
    print("Analysis complete.")