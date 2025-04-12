from vectorDB.vectorDBingestion import VectorDBIngestion
from evaluation.evaluation import Evaluation

vectorDBingestion : VectorDBIngestion = VectorDBIngestion()
evaluation : Evaluation = Evaluation()

vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoERafL2_32_16", True)
vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoERafL2_64_16", True)
vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoERafL2_128_16", True)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="ET",
    collection="moiraiMoERafL2_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="ET",
    collection="moiraiMoERafL2_64_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="ET",
    collection="moiraiMoERafL2_128_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="power",
    collection="moiraiMoERafL2_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="power",
    collection="moiraiMoERafL2_64_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="power",
    collection="moiraiMoERafL2_128_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="m4-monthly",
    collection="moiraiMoERafL2_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="m4-monthly",
    collection="moiraiMoERafL2_64_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="m4-monthly",
    collection="moiraiMoERafL2_128_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="traffic",
    collection="moiraiMoERafL2_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="traffic",
    collection="moiraiMoERafL2_64_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="traffic",
    collection="moiraiMoERafL2_128_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="huaweiCloud",
    collection="moiraiMoERafL2_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="huaweiCloud",
    collection="moiraiMoERafL2_64_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="huaweiCloud",
    collection="moiraiMoERafL2_128_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="electricityUCI",
    collection="moiraiMoERafL2_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="electricityUCI",
    collection="moiraiMoERafL2_64_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="electricityUCI",
    collection="moiraiMoERafL2_128_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="covid19Deaths",
    collection="moiraiMoERafL2_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="covid19Deaths",
    collection="moiraiMoERafL2_64_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="covid19Deaths",
    collection="moiraiMoERafL2_128_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="fredMd",
    collection="moiraiMoERafL2_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="fredMd",
    collection="moiraiMoERafL2_64_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="fredMd",
    collection="moiraiMoERafL2_128_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="nn5",
    collection="moiraiMoERafL2_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="nn5",
    collection="moiraiMoERafL2_64_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="nn5",
    collection="moiraiMoERafL2_128_16",
)

report : dict = evaluation.compileReports(reportOriginName="evaluationReportsMoiraiMoERafSoftMax",reportTargetName="evaluationFinalReportMoiraiMoERafSoftMax")
