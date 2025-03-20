from vectorDB.vectorDBingestion import VectorDBIngestion
from evaluation.evaluation import Evaluation

vectorDBingestion : VectorDBIngestion = VectorDBIngestion()
evaluation : Evaluation = Evaluation()

vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoECosine_32_16")
vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoECosine_64_16")
vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoECosine_128_16")

report : dict = evaluation.evaluateMoiraiMoERag(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="ET",
    collection="moiraiMoECosine_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERag(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="ET",
    collection="moiraiMoECosine_64_16",
)

report : dict = evaluation.evaluateMoiraiMoERag(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="ET",
    collection="moiraiMoECosine_128_16",
)

report : dict = evaluation.evaluateMoiraiMoERag(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="solarEnergy",
    collection="moiraiMoECosine_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERag(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="power",
    collection="moiraiMoECosine_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERag(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="power",
    collection="moiraiMoECosine_64_16",
)

report : dict = evaluation.evaluateMoiraiMoERag(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="power",
    collection="moiraiMoECosine_128_16",
)

report : dict = evaluation.evaluateMoiraiMoERag(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="m4-monthly",
    collection="moiraiMoECosine_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERag(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="m4-monthly",
    collection="moiraiMoECosine_64_16",
)

report : dict = evaluation.evaluateMoiraiMoERag(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="m4-monthly",
    collection="moiraiMoECosine_128_16",
)

report : dict = evaluation.compileReports(reportOriginName="evaluationReportsMoiraiMoERag",reportTargetName="evaluationFinalReportRag")