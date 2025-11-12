from vectorDB.vectorDBingestion import VectorDBIngestion
from evaluation.evaluation import Evaluation

vectorDBingestion : VectorDBIngestion = VectorDBIngestion()
evaluation : Evaluation = Evaluation()

vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoECosine_32_16")
vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoECosine_64_16")
vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoECosine_128_16")

vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoEL2_32_16")
vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoEL2_64_16")
vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoEL2_128_16")

vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoERafL2_32_16", raf=True)
vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoERafL2_64_16", raf=True)
vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoERafL2_128_16", raf=True)

report : dict = evaluation.evaluateMoiraiMoERagCA(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="ET",
    collection="moiraiMoERafL2_64_16",
    loadPretrainedRagCA=False,
)

report : dict = evaluation.evaluateMoiraiMoERagCA(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="electricityUCI",
    collection="moiraiMoERafL2_64_16",
    loadPretrainedRagCA=False,
)

report : dict = evaluation.evaluateMoiraiMoERagCA(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="huaweiCloud",
    collection="moiraiMoERafL2_64_16",
    loadPretrainedRagCA=False,
)

report : dict = evaluation.evaluateMoiraiMoERagCA(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="power",
    collection="moiraiMoERafL2_64_16",
    loadPretrainedRagCA=False,
)

report : dict = evaluation.evaluateMoiraiMoERagCA(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="traffic",
    collection="moiraiMoERafL2_64_16",
    loadPretrainedRagCA=False,
)