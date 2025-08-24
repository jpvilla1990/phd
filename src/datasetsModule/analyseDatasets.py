import random
import concurrent.futures
import pandas as pd
import numpy as np
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

if __name__ == "__main__":
    dataset = "ET"
    contextLength = 32
    predictionLength = 16
    element = "value"
    
    analyser = AnalyseDataset()
    analyser.loadSubset(dataset, contextLength, predictionLength, trainSet=False)
    print("Analysis complete.")