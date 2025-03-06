from typing import Callable
import uuid
import numpy as np
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from utils.fileSystem import FileSystem

class CustomEmbeddingFunction(object):
    """
    Class to handle custom embedding functions
    """
    def __init__(self, embedingFunction : Callable):
        super().__init__()
        self.__embedingFunction : Callable = embedingFunction

    def __call__(self, input : np.ndarray):
        return self.__embedingFunction(input)

class vectorDB(FileSystem):
    """
    Class to handle vector database
    """
    def __init__(self):
        super().__init__()
        self.__chromaClient : chromadb.api.client.Client = chromadb.PersistentClient(
            path=self._getPaths()["vectorDatabase"],
            settings=Settings(anonymized_telemetry=False)
        )

    def setCollection(
            self,
            collection : str,
            embeddingFunction : Callable = None,
        ):
        """
        Method to set collection
        """
        try:
            self.__collection : chromadb.api.models.Collection.Collection = self.__chromaClient.get_collection(
                name=collection,
            )
        except:
            self.__collection : chromadb.api.models.Collection.Collection = self.__chromaClient.create_collection(
                name=collection,
                embedding_function=CustomEmbeddingFunction(embeddingFunction),
                metadata=self._getConfig()["vectorDatabase"]["distance"]["euclidean"]
            )

    def ingestTimeseries(self, context : np.ndarray, prediction : np.ndarray):
        """
        Method to ingest time series in collection
        """
        self.__collection.add(
            ids=[str(uuid.uuid4)],
            embeddings=[context],
            metadatas=[{
                "context" : ",".join(map(str, context.tolist())),
                "prediction" : ",".join(map(str, prediction.tolist())),
            }]
        )