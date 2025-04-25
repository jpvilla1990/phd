from vectorDB.vectorDBingestion import VectorDBIngestion
from evaluation.evaluation import Evaluation

vectorDBingestion : VectorDBIngestion = VectorDBIngestion()
evaluation : Evaluation = Evaluation()

vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoECosine_128_16")
vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoECosine_64_16")
vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoECosine_32_16")

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="ET",
    collection="moiraiMoECosine_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="ET",
    collection="moiraiMoECosine_64_16",
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="ET",
    collection="moiraiMoECosine_128_16",
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="power",
    collection="moiraiMoECosine_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="power",
    collection="moiraiMoECosine_64_16",
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="power",
    collection="moiraiMoECosine_128_16",
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="m4-monthly",
    collection="moiraiMoECosine_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="m4-monthly",
    collection="moiraiMoECosine_64_16",
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="m4-monthly",
    collection="moiraiMoECosine_128_16",
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="traffic",
    collection="moiraiMoECosine_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="traffic",
    collection="moiraiMoECosine_64_16",
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="traffic",
    collection="moiraiMoECosine_128_16",
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="huaweiCloud",
    collection="moiraiMoECosine_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="huaweiCloud",
    collection="moiraiMoECosine_64_16",
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="huaweiCloud",
    collection="moiraiMoECosine_128_16",
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="electricityUCI",
    collection="moiraiMoECosine_128_16",
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="electricityUCI",
    collection="moiraiMoECosine_64_16",
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="electricityUCI",
    collection="moiraiMoECosine_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="covid19Deaths",
    collection="moiraiMoECosine_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="covid19Deaths",
    collection="moiraiMoECosine_64_16",
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="covid19Deaths",
    collection="moiraiMoECosine_128_16",
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="fredMd",
    collection="moiraiMoECosine_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="fredMd",
    collection="moiraiMoECosine_64_16",
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="fredMd",
    collection="moiraiMoECosine_128_16",
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="nn5",
    collection="moiraiMoECosine_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="nn5",
    collection="moiraiMoECosine_64_16",
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="nn5",
    collection="moiraiMoECosine_128_16",
)

report : dict = evaluation.compileReports(reportOriginName="evaluationReportsMoiraiMoERagSoftMax",reportTargetName="evaluationFinalReportMoiraiMoERagSoftMax")