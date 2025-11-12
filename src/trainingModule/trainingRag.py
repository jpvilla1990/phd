import pandas as pd
from vectorDB.vectorDB import vectorDB
from model.moiraiMoe import MoiraiMoE

class TrainingRag(object):
    def __init__(
        self,
        collection : str,
        dataset : str,
        contextLength : int,
        predictionLength : int,
        numberSamples : int,
    ):
        print(f"TrainingRag collection: {collection}, dataset: {dataset}, contextLength: {contextLength}, predictionLength: {predictionLength}")
        self.__vectorDB : vectorDB = vectorDB()
        self.__vectorDB.setCollection(
            f"{collection}_{contextLength}_{predictionLength}",
            dataset,
            lambda x : torch.tensor(x).reshape(1, len(x)),
        )
        self.numberSamples : int = numberSamples
        self.contextLength : int = contextLength
        self.predictionLength : int = predictionLength
        self.dataset : str = dataset

    def train(self, force : bool = False):
        """
        Method to train the RAG dataset to set when it should augment and when not.
        """
        ids : list = self.__vectorDB.getAllCollection()["ids"]

        force = True

        for id in ids:
            sample : dict = self.__vectorDB.getSample(id)
            metadata : dict = sample["metadatas"][0]
            if not force:
                if "trained" in metadata:
                    if metadata["trained"]:
                        continue

            if "augmentation" not in metadata:
                metadata["augmentation"] = 1.0

            model : MoiraiMoE = MoiraiMoE(
                predictionLength = self.predictionLength,
                contextLength = self.contextLength,
                numSamples = self.numberSamples,
            )
            modelAugmented : MoiraiMoE = MoiraiMoE(
                predictionLength =self.predictionLength,
                contextLength = self.contextLength + int((self.predictionLength + self.contextLength) * metadata["augmentation"]),
                numSamples = int(self.numberSamples),
            )

            timeseries : list = [float(x) for x in sample["documents"][0].split(",")]
            groundTruth : list = [float(x) for x in sample["metadatas"][0]["prediction"].split(",")]
            augmented : list = timeseries + groundTruth + timeseries

            startTime = pd.Timestamp.today()
            samplingInterval = pd.Timedelta(milliseconds=1000 * 60)

            df = pd.DataFrame({
                "timestamp": [startTime + i * samplingInterval for i in range(len(timeseries))],
                "sample": timeseries
            })

            dfAugmented = pd.DataFrame({
                "timestamp": [startTime + i * samplingInterval for i in range(len(augmented))],
                "sample": augmented
            })

            prediction = model.inference(df, self.dataset)
            predictionAugmented = modelAugmented.inference(dfAugmented, self.dataset)

            mse = ((pd.Series(prediction) - pd.Series(groundTruth)) ** 2).mean()
            mseAugmented = ((pd.Series(predictionAugmented) - pd.Series(groundTruth)) ** 2).mean()

            if mse <= mseAugmented:
                metadata["augmentation"] = 0.0

            metadata["trained"] = True

            self.__vectorDB.updateMetadata([id], [metadata])

if __name__ == "__main__":
    training : TrainingRag = TrainingRag(
        collection="moiraiMoERafL2",
        dataset="ET",
        predictionLength=16,
        contextLength=32,
        numberSamples=100,
    )
    training.train()