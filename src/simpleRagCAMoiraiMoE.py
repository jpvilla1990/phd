from vectorDB.vectorDBingestion import VectorDBIngestion
from training.training import Training

vectorDBingestion : VectorDBIngestion = VectorDBIngestion()
training : Training = Training()

#vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoECosine_32_16")
#vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoECosine_64_16")
#vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoECosine_128_16")

report : dict = training.trainRagCA(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="ET",
)
