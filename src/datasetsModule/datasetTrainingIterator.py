import pandas as pd
import torch
from torch.utils.data import Dataset
import random
from datasetsModule.datasets import Datasets
from datasetsModule.datasetIterator import DatasetIterator

class DatasetTrainingIterator(Dataset):
    def __init__(
            self,
            config : dict,
            datasetsConfig : dict,
        ):
        self.__config : dict = config
        self.__dataset : str = config["training"]["dataset"]
        Datasets().loadDataset(self.__dataset)
        self.__subdatasets = {key : None for key in list(datasetsConfig[self.__dataset].keys())}
        self.__batchSize : int = config["training"]["batchSize"]
        self.__maxIterationsPerEpoch : int = config["training"]["maxIterationsPerEpoch"]

        random.seed(config["seed"])
        self.__iterators = self.getIterators(config["training"]["lengthCombinations"])


    def __resetIterators(self):
        """
        Reset the iterators for the dataset.
        """
        for key in list(self.__iterators.keys()):
            for element in self.__subdatasets.keys():
                self.__iterators[key].resetIteration(element, True, trainPartition=self.__config["trainPartition"])

    def getIterators(self, combinations : list) -> dict:
        """
        Get the iterators for the dataset.
        """
        iterators : dict = {}
        for combination in combinations:
            contextLength : int = combination["contextLength"]
            predictionLength : int = combination["predictionLength"]
            iterator : DatasetIterator = Datasets().loadDataset(self.__dataset)
            iterator.setSampleSize(contextLength + predictionLength)
            for element in self.__subdatasets.keys():
                if self.__subdatasets[element] is None:
                    self.__subdatasets[element] = list(iterator.getAvailableFeatures(element).keys())
                iterator.resetIteration(element, True, trainPartition=self.__config["trainPartition"])
            iterators[f"{contextLength}_{predictionLength}"] = iterator
        return iterators

    def __getitem__(self, idx : int) -> tuple[torch.Tensor, int]:
        """
        Get a random sample from the dataset.
        This method retrieves a random sample from the dataset, ensuring that the sample is valid and does not contain any NaN values.
        """
        samples : torch.Tensor = None  

        running : bool = True
        batchIndex : int = 0

        combination : str = random.choice(list(self.__iterators.keys()))
        iterator : DatasetIterator = self.__iterators[combination]
        contextLength : int = int(combination.split("_")[0])
        predictionLength : int = int(combination.split("_")[1])

        if idx == 0:
            self.__resetIterators()

        while running:
            subdataset : str = random.choice(list(self.__subdatasets.keys()))
            features : list = self.__subdatasets[subdataset]
            try:
                sample : pd.core.frame.DataFrame = iterator.iterateDataset(
                    subdataset,
                    self.__subdatasets[subdataset],
                    True,
                )
                if sample is None:
                    continue
                if len(sample) < predictionLength + contextLength:
                    continue

                indexes : list = [index for index in range(1,len(features))]
                random.shuffle(indexes)
                for i in range(len(indexes)):
                    index : int = indexes[i]
                    if sample[index].isna().any().any():
                        continue

                    torchSample : torch.Tensor = torch.tensor(
                        sample[[index]].to_numpy(),
                        dtype=torch.float32,
                    ).permute(1,0)

                    if batchIndex >= self.__batchSize:
                        running = False
                    else:
                        if samples is None:
                            samples = torchSample
                        else:
                            samples = torch.cat((samples, torchSample), dim=0)

                    batchIndex += 1
            except Exception as e:
                raise e
                print("Exception: " + str(e))
                continue

        return samples, contextLength

    def __len__(self):
        return self.__maxIterationsPerEpoch