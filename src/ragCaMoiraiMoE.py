from vectorDB.vectorDBingestion import VectorDBIngestion
from trainingModule.training import Training

vectorDBingestion : VectorDBIngestion = VectorDBIngestion()

vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoETrainingCosine_32_16", train=False)
vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoETrainingCosine_64_16", train=False)
vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoETrainingCosine_128_16", train=False)

training :Training = Training()
training.trainRagCA()