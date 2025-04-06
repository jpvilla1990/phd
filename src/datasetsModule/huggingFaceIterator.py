

class HuggingFaceIterator(object):
    """
    Class to handle iterators on the datasets
    """
    def __init__(self, name : str, datasets : dict, datasetConfig : dict, seed : int = 42):
        random.seed(seed)
        self.__datasets : dict = datasets
        self.__datasetConfig : dict = datasetConfig
        self.__name : str = name

        self.__features : dict = {}
        self.__datasetSizes : dict = {}

    def iterateDataset(
            self,
            train : bool = True,
        ) -> pd.core.frame.DataFrame:
        """
        Method to iterate through out the whole dataset
        """
        category : str = "test"
        if train:
            category = "train"

        if len(self.__indexIterator[subdataset][category]) == 0:
            if "frame" in self.__indexIterator[subdataset]:
                del self.__indexIterator[subdataset]["frame"]
            return None

        sample : pd.core.frame.Dataframe = self.loadSample(
            subdataset=subdataset,
            sampleIndex=self.__indexIterator[subdataset][category][-1],
            sampleSize=self.__sampleSize,
            features=features,
        )

        self.__indexIterator[subdataset][category].pop()

        return sample