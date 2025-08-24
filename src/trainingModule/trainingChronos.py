import os
import concurrent.futures
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from utils.fileSystem import FileSystem
from utils.utils import Utils
from model.moiraiMoe import MoiraiMoE
from model.chronosModel import Chronos
from datasetsModule.datasetTrainingIterator import DatasetTrainingIterator
from model.ragCrossAttention import RagCrossAttention
from utils.utils import Utils

class TrainingRagCA(L.LightningModule):
    def __init__(
        self,
        dataset : str,
        collectionPrefix : str,
        k : int = 16,
        lr : float = 1e-3,
        numberSamples : int = 100,
        batchSize : int = 1,
        weightDecay : float = 1e-1,
        betas : tuple = (0.9, 0.98),
        eps : float = 1e-6,
        loadPretrainedModel : bool = False,
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
        self.__loadPretrainedModel : bool = loadPretrainedModel

        self.modelRagCA : RagCrossAttention = RagCrossAttention(
            loadPretrainedModel=self.__loadPretrainedModel,
        )

        self.backBoneModels : Chronos = Chronos(
            bolt=False,
            frozen=True,
        )

    def training_step(
            self,
            batch : tuple[torch.Tensor, int],
            batch_idx : int,
        ) -> torch.Tensor:
        """
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

        xContext : torch.Tensor = batch[:, :contextLength]
        xTarget : torch.Tensor = batch[:, contextLength:]

        self.backBoneModels.setRafCollection(f"{self.__collectionPrefix}_{contextLength}_{predictionLength}", self.__dataset)

        queried, scores =  self.backBoneModels.queryBatchVector(xContext, k=self.__k)

        augmentedSample, mean, std = self.modelRagCA.forward(
            xContext,
            queried,
            scores,
        )

        loss = self.backBoneModels.forwardLoss(augmentedSample, mean, std, xTarget)

        print(loss)

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
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

    def __writeReport(self, enztry : dict, report : str):
        """
        Method to write in dataset config
        """
        Utils.writeYaml(
            self._getFiles()[report],
            self.__loadReport(report) | entry,
        )

    def saveModelState(self, parametersFile : str):
        """
        Method to get RAG CA state
        """
        parametersFile = os.path.join(self._getPaths()["RagCAModels"], parametersFile)
        print(parametersFile)
        model : RagCrossAttention = TrainingRagCA.load_from_checkpoint(
            parametersFile,
            dataset=self._getConfig()["training"]["dataset"],
            collectionPrefix=self._getConfig()["training"]["collectionPrefix"],
            k=self._getConfig()["training"]["k"],
            batchSize=self.__batchSize,
            lr=float(self._getConfig()["training"]["lr"]),
            weightDecay=float(self._getConfig()["training"]["weightDecay"]),
            betas=(float(self._getConfig()["training"]["beta1"]), float(self._getConfig()["training"]["beta2"])),
            eps=float(self._getConfig()["training"]["eps"]),
        ).modelRagCA
        torch.save(model.state_dict(), self._getFiles()["paramsRagCA"])

    def trainRagCA(
            self,
            modelName : str,
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
            balanced=True,
        )
        tsLoader = DataLoader(tsDataset, batch_size=1, num_workers=0)
        trainer = L.Trainer(
            max_epochs=self.__epochs,
            accelerator="auto",
            log_every_n_steps=1,
            callbacks=[
                ModelCheckpoint(
                    dirpath=self._getPaths()["RagCAModels"],
                    filename=f"RagCA-{modelName}-{self._getConfig()['training']['dataset']}-{{epoch:02d}}-{{step:06d}}",
                    save_top_k=-1,
                    every_n_train_steps=1000,
                    save_on_train_epoch_end=False,
                ),
            ],
        )
        if self._getConfig()["training"]["checkpoint"] != "":
            print(self._getConfig()["training"]["checkpoint"])
            trainer.fit(
                model,
                tsLoader,
                ckpt_path = os.path.join(
                    self._getPaths()["RagCAModels"],
                    self._getConfig()["training"]["checkpoint"],
                ),
            )
        else:
            trainer.fit(
                model,
                tsLoader, 
            )

    def test(self):
        tsDataset = DatasetTrainingIterator(
            self._getConfig(),
            Utils.readYaml(self._getFiles()["datasets"]),
            balanced=True,
        )
        a = tsDataset.test(0)

if __name__ == "__main__":
    training : Training = Training()
    #training.trainRagCA()