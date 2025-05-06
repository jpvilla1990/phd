import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from utils.fileSystem import FileSystem
from utils.utils import Utils
from model.moiraiMoe import MoiraiMoE
from datasetsModule.datasetTrainingIterator import DatasetTrainingIterator
from model.ragCrossAttention import RagCrossAttention

class TrainingRagCA(L.LightningModule):
    def __init__(
        self,
        dataset : str,
        collectionPrefix : str,
        k : int = 1,
        lr : float = 1e-3,
        numberSamples : int = 100,
        batchSize : int = 1,
        weightDecay : float = 1e-1,
        betas : tuple = (0.9, 0.98),
        eps : float = 1e-6,
    ):
        super().__init__()
        self.__dataset : str = dataset
        self.__collectionPrefix : str = collectionPrefix
        self.__k : int = k
        self.__lr : float = lr
        self.numberSamples : int = numberSamples
        self.__batchSize : int = batchSize
        self.__weightDecay : float = weightDecay
        self.__betas : tuple = betas
        self.__eps : float = eps

        self.modelRagCA : RagCrossAttention = RagCrossAttention()

        self.backBoneModels : dict = {}

    def getBackBoneModel(self, contextLength : int, predictionLength : int) -> MoiraiMoE:
        """
        Method to get the backbone model
        :return: Backbone model
        """
        index : str = f"{contextLength}_{predictionLength}"
        if index not in self.backBoneModels:
            self.backBoneModels[index] = MoiraiMoE(
                predictionLength = predictionLength,
                contextLength = contextLength,
                numSamples = self.numberSamples,
                batchSize = self.__batchSize,
                createPredictor=False,
                frozen=True,
            )
            self.backBoneModels[index].setRagCollection(f"{self.__collectionPrefix}_{index}", self.__dataset)
        return self.backBoneModels[index]

    def training_step(
            self,
            batch : tuple[torch.Tensor, int],
            batch_idx : int,
        ) -> torch.Tensor:
        """s
        Training step for the model.
        :param batch: Batch of data to be used for training. [B, L]
        :return: Loss value for the batch.
        """
        print(f"Batch: {batch_idx}")
        batch, contextLength = batch
        if contextLength.shape == torch.Size([1]):
            contextLength = contextLength.item()
        else:
            contextLength = contextLength[-1].item()
        batch = batch.squeeze(0)
        predictionLength : int = batch.shape[1] - contextLength
        modelBackBone : MoiraiMoE = self.getBackBoneModel(
            contextLength = contextLength,
            predictionLength = predictionLength,
        )

        xContext : torch.Tensor = batch[:, :contextLength]
        xTarget : torch.Tensor = batch[:, contextLength:]

        queried, scores = modelBackBone.queryBatchVector(xContext, k=self.__k, metadata={"dataset": self.__dataset})

        augmentedSample : torch.Tensor = self.modelRagCA.forward(
            xContext,
            xContext,
            torch.randn(1, 1, 1).to(xContext.device),
        )

        pred = modelBackBone.forwardRagCA(
            augmentedSample,
        )

        xTarget = xTarget.unsqueeze(1)
        loss = -pred.log_prob(xTarget)
        loss = loss.mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        print(self.__lr)
        print(type(self.__lr))
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.__lr,
            weight_decay=self.__weightDecay,
            betas=self.__betas,
            eps=self.__eps,
        )

class Training(FileSystem):
    """
    Class to evaluate models
    """
    def __init__(self):
        super().__init__()
        self.__batchSize : int = self._getConfig()["training"]["batchSize"]
        self.__epochs : int = self._getConfig()["training"]["epochs"]

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

    def trainRagCA(
            self,
        ):
        """
        Method to train the RagCA model
        """
        model : TrainingRagCA = TrainingRagCA(
            dataset=self._getConfig()["training"]["dataset"],
            collectionPrefix=self._getConfig()["training"]["collectionPrefix"],
            k=self._getConfig()["training"]["k"],
            batchSize=self.__batchSize,
            lr=float(self._getConfig()["training"]["lr"]),
            weightDecay=float(self._getConfig()["training"]["weightDecay"]),
            betas=(float(self._getConfig()["training"]["beta1"]), float(self._getConfig()["training"]["beta2"])),
            eps=float(self._getConfig()["training"]["eps"]),
        )
        tsDataset = DatasetTrainingIterator(
            self._getConfig(),
            Utils.readYaml(self._getFiles()["datasets"]),
        )
        tsLoader = DataLoader(tsDataset, batch_size=1, num_workers=0)
        trainer = L.Trainer(
            max_epochs=self.__epochs,
            accelerator="auto",
            log_every_n_steps=1,
            callbacks=[
                ModelCheckpoint(
                    monitor="train_loss",
                    dirpath=self._getPaths()["RagCAModels"],
                    filename=f"RagCA-{self._getConfig()['training']['dataset']}-{{epoch:02d}}-{{train_loss:.2f}}",
                    save_top_k=1,
                    mode="min",
                    every_n_train_steps=1000,
                    save_weights_only=False,
                ),
            ]
        )
        trainer.fit(model, tsLoader)

if __name__ == "__main__":
    training = Training()
    training.trainRagCA()