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
    collection="moiraiMoERaf_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="ET",
    collection="moiraiMoERaf_64_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="ET",
    collection="moiraiMoERaf_128_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="power",
    collection="moiraiMoERaf_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="power",
    collection="moiraiMoERaf_64_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="power",
    collection="moiraiMoERaf_128_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="m4-monthly",
    collection="moiraiMoERaf_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="m4-monthly",
    collection="moiraiMoERaf_64_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="m4-monthly",
    collection="moiraiMoERaf_128_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="traffic",
    collection="moiraiMoERaf_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="traffic",
    collection="moiraiMoERaf_64_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="traffic",
    collection="moiraiMoERaf_128_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="huaweiCloud",
    collection="moiraiMoERaf_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="huaweiCloud",
    collection="moiraiMoERaf_64_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="huaweiCloud",
    collection="moiraiMoERaf_128_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="electricityUCI",
    collection="moiraiMoERaf_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="electricityUCI",
    collection="moiraiMoERaf_64_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="electricityUCI",
    collection="moiraiMoERaf_128_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="covid19Deaths",
    collection="moiraiMoERaf_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="covid19Deaths",
    collection="moiraiMoERaf_64_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="covid19Deaths",
    collection="moiraiMoERaf_128_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="fredMd",
    collection="moiraiMoERaf_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="fredMd",
    collection="moiraiMoERaf_64_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="fredMd",
    collection="moiraiMoERaf_128_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="nn5",
    collection="moiraiMoERaf_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="nn5",
    collection="moiraiMoERaf_64_16",
)

report : dict = evaluation.evaluateMoiraiMoERafSoftMax(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="nn5",
    collection="moiraiMoERaf_128_16",
)

report : dict = evaluation.compileReports(reportOriginName="evaluationReportsMoiraiMoERafSoftMax",reportTargetName="evaluationFinalReportMoiraiMoERafSoftMax")
