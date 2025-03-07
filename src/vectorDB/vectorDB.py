from typing import Callable
import uuid
import numpy as np
import chromadb
from chromadb.config import Settings
from chromadb import Documents, EmbeddingFunction, Embeddings
from utils.fileSystem import FileSystem

class CustomEmbeddingFunction(EmbeddingFunction):
    """
    Class to handle custom embedding functions
    """
    def __init__(self, embedingFunction : Callable):
        super().__init__()
        self.__embedingFunction : Callable = embedingFunction

    def __call__(self, input : Documents) -> Embeddings:
        inputArray : np.ndarray = np.array([float(element) for element in input[0].split(",")])
        return self.__embedingFunction(inputArray).numpy()[0]

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
        self.__collection : chromadb.api.models.Collection.Collection = self.__chromaClient.get_or_create_collection(
            name=collection,
            embedding_function=CustomEmbeddingFunction(embeddingFunction),
            metadata=self._getConfig()["vectorDatabase"]["distance"]["euclidean"]
        )

    def ingestTimeseries(self, context : np.ndarray, prediction : np.ndarray):
        """
        Method to ingest time series in collection
        """
        contextStr : str = ",".join(map(str, context.tolist()))
        predictionStr : str = ",".join(map(str, prediction.tolist()))
        id = str(uuid.uuid4())
        self.__collection.add(
            ids=[id],
            documents=[contextStr],
            metadatas=[{
                "prediction" : predictionStr,
            }]
        )

    def queryTimeseries(self, query : np.ndarray, k : int = 1) -> list:
        """
        Method to query element from vector database

        return list[context + prediction]
        """
        queryStr : str = ",".join(map(str, query.tolist()))
        queried = self.__collection.query(
            n_results=k,
            query_texts=[queryStr]
        )
        documents : list = queried["documents"][0]
        metadatas : list = queried["metadatas"][0]

        predictions : list = [metadata["prediction"] for metadata in metadatas]

        output : list = [f"{documents[index]},{predictions[index]}" for index in range(len(documents))]

        return [np.array([float(sample) for sample in element.split(",")]) for element in output]