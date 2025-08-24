from vectorDB.vectorDBingestion import VectorDBIngestion
from trainingModule.trainingChronos import Training

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
#training.saveModelState("RagCA-ragCA-lotsaData-epoch=00-step=000900-v1.ckpt")