import torch
import lightning as L
from torch.utils.data import DataLoader

from utils.fileSystem import FileSystem
from utils.utils import Utils
from model.moiraiMoe import MoiraiMoE
from datasetsModule.datasetTrainingIterator import DatasetTrainingIterator
from model.ragCrossAttention import RagCrossAttention

class TrainingRagCA(L.LightningModule):
    def __init__(self, lr=1e-3, numberSamples=100):
        super().__init__()
        self.lr : float = lr
        self.numberSamples : int = numberSamples

        self.modelRagCA : RagCrossAttention = RagCrossAttention()

        self.backBoneModels : dict = {}

        for param in self.model.parameters():
            param.requires_grad = False

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
            )
            for param in self.backBoneModels[index].parameters():
                param.requires_grad = False
        return self.backBoneModels[index]

    def training_step(
            self,
            batch : torch.Tensor,
            contextLength : int,
            predictionLength : int,
        ) -> torch.Tensor:
        """
        Training step for the model.
        :param batch: Batch of data to be used for training. [B, L]
        :return: Loss value for the batch.
        """
        modelBackBone : MoiraiMoE = self.getBackBoneModel(
            contextLength = contextLength,
            predictionLength = predictionLength,
        )

        xContext : torch.Tensor = batch[:, :contextLength]
        xTarget : torch.Tensor = batch[:, contextLength:]

        augmentedSample : torch.Tensor = self.modelRagCA.forward(
            xContext,
            xContext,
            torch.randn(1, 1, 1),
        )

        pred = modelBackBone.forwardRagCA(
            augmentedSample,
        )

        loss : torch.Tensor = -pred.log_prob(xTarget)
        loss = loss.mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

class Training(FileSystem):
    """
    Class to evaluate models
    """
    def __init__(self):
        super().__init__()

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
        model : TrainingRagCA = TrainingRagCA()
        tsDataset = DatasetTrainingIterator(
            self._getConfig(),
            Utils.readYaml(self._getFiles()["datasets"]),
        )
        tsLoader = DataLoader(tsDataset, batch_size=8, num_workers=0)
        trainer = L.Trainer(max_epochs=10, accelerator="auto", log_every_n_steps=1)
        trainer.fit(model, tsLoader)

if __name__ == "__main__":
    training = Training()
    training.trainRagCA()