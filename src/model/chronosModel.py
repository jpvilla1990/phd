import numpy as np
import torch
from utils.fileSystem import FileSystem
from chronos import BaseChronosPipeline
from vectorDB.vectorDB import vectorDB
from model.ragCrossAttention import RagCrossAttention

class Chronos(FileSystem):
    """
    Datasets used pretrained:
    electricityUCI
    m4
    nn5
    fredmd
    """
    def __init__(
        self,
        model : str = "amazon/chronos-bolt-small",
    ):
        super().__init__()
        self.__model : BaseChronosPipeline = BaseChronosPipeline.from_pretrained(
            model,
            device_map = "cuda" if torch.cuda.is_available() else "cpu",
        )
        self.__vectorDB : vectorDB = vectorDB()
        self.__modelRagCA : RagCrossAttention = RagCrossAttention(
            patchSize=16,
            pretrainedModel=self._getFiles()["paramsRagCA"],
            loadPretrainedModel=True,
        )

    def setRafCollection(self, collectionName : str, dataset : str):
        """
        Method to set RAF collection
        """
        self.__vectorDB.setCollection(
            collectionName,
            dataset,
            lambda x : torch.tensor(x).reshape(1, len(x)),
        )

    def queryVector(self, sample : np.ndarray, k : int = 1) -> tuple:
        """
        Method to query vector
        """
        return self.__vectorDB.queryTimeseries(sample, k)

    def predict(self, sample : np.ndarray | torch.Tensor, predictionLength : int = 16) -> np.ndarray:
        sample = torch.tensor(sample) if type(sample) == np.ndarray else sample
        return self.__model.predict_quantiles(
            context=sample,
            prediction_length=predictionLength,
        )[1].squeeze().numpy()

    def predictRag(self, sample : np.ndarray | torch.Tensor, predictionLength : int = 16) -> np.ndarray:
        query : torch.tensor = torch.tensor(sample, dtype=torch.float32)
        queried, score = self.queryVector(query, k=16)
        if queried is not None:
            xContext : torch.Tensor = query.unsqueeze(0)
            queriedTorch : torch.Tensor = torch.Tensor(queried).unsqueeze(0)
            scoreTensor : torch.Tensor = torch.Tensor(score).unsqueeze(0)

            augmentedSample : torch.Tensor = self.__modelRagCA.inference(
                xContext,
                queriedTorch,
                scoreTensor,
            ).squeeze()

            return self.predict(
                augmentedSample,
                predictionLength,
            )
        else:
            return self.predict(
                sample,
                predictionLength,
            )