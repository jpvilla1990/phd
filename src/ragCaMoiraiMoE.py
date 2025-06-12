from vectorDB.vectorDBingestion import VectorDBIngestion
from trainingModule.training import Training

vectorDBingestion : VectorDBIngestion = VectorDBIngestion()

vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoETrainingCosine_32_16", train=False)
vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoETrainingCosine_64_16", train=False)
vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoETrainingCosine_128_16", train=False)

vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoETrainingL2_32_16", train=False)
vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoETrainingL2_64_16", train=False)
vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoETrainingL2_128_16", train=False)

vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoETrainingRafL2_32_16", raf=True, train=False)
vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoETrainingRafL2_64_16", raf=True, train=False)
vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoETrainingRafL2_128_16", raf=True, train=False)

training :Training = Training()
training.trainRagCA("ragCA")
#training.saveModelState("RagCA-lotsaData-epoch=00-step=4046-train_loss=-3.19.ckpt")