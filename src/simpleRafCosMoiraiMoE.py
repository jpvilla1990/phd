from vectorDB.vectorDBingestion import VectorDBIngestion
from evaluation.evaluation import Evaluation

vectorDBingestion : VectorDBIngestion = VectorDBIngestion()
evaluation : Evaluation = Evaluation()

vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoERafNormCos_32_16", True)
vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoERafNormCos_64_16", True)
vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoERafNormCos_128_16", True)

report : dict = evaluation.evaluateMoiraiMoERafCosSoftMax(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="ET",
    collection="moiraiMoERafNormCos_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERafCosSoftMax(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="ET",
    collection="moiraiMoERafNormCos_64_16",
)

report : dict = evaluation.evaluateMoiraiMoERafCosSoftMax(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="ET",
    collection="moiraiMoERafNormCos_128_16",
)

report : dict = evaluation.evaluateMoiraiMoERafCosSoftMax(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="power",
    collection="moiraiMoERafNormCos_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERafCosSoftMax(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="power",
    collection="moiraiMoERafNormCos_64_16",
)

report : dict = evaluation.evaluateMoiraiMoERafCosSoftMax(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="power",
    collection="moiraiMoERafNormCos_128_16",
)

report : dict = evaluation.evaluateMoiraiMoERafCosSoftMax(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="m4-monthly",
    collection="moiraiMoERafNormCos_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERafCosSoftMax(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="m4-monthly",
    collection="moiraiMoERafNormCos_64_16",
)

report : dict = evaluation.evaluateMoiraiMoERafCosSoftMax(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="m4-monthly",
    collection="moiraiMoERafNormCos_128_16",
)

report : dict = evaluation.evaluateMoiraiMoERafCosSoftMax(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="traffic",
    collection="moiraiMoERafNormCos_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERafCosSoftMax(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="traffic",
    collection="moiraiMoERafNormCos_64_16",
)

report : dict = evaluation.evaluateMoiraiMoERafCosSoftMax(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="traffic",
    collection="moiraiMoERafNormCos_128_16",
)

report : dict = evaluation.evaluateMoiraiMoERafCosSoftMax(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="huaweiCloud",
    collection="moiraiMoERafNormCos_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERafCosSoftMax(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="huaweiCloud",
    collection="moiraiMoERafNormCos_64_16",
)

report : dict = evaluation.evaluateMoiraiMoERafCosSoftMax(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="huaweiCloud",
    collection="moiraiMoERafNormCos_128_16",
)

report : dict = evaluation.evaluateMoiraiMoERafCosSoftMax(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="electricityUCI",
    collection="moiraiMoERafNormCos_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERafCosSoftMax(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="electricityUCI",
    collection="moiraiMoERafNormCos_64_16",
)

report : dict = evaluation.evaluateMoiraiMoERafCosSoftMax(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="electricityUCI",
    collection="moiraiMoERafNormCos_128_16",
)

report : dict = evaluation.evaluateMoiraiMoERafCosSoftMax(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="covid19Deaths",
    collection="moiraiMoERafNormCos_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERafCosSoftMax(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="covid19Deaths",
    collection="moiraiMoERafNormCos_64_16",
)

report : dict = evaluation.evaluateMoiraiMoERafCosSoftMax(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="covid19Deaths",
    collection="moiraiMoERafNormCos_128_16",
)

report : dict = evaluation.evaluateMoiraiMoERafCosSoftMax(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="fredMd",
    collection="moiraiMoERafNormCos_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERafCosSoftMax(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="fredMd",
    collection="moiraiMoERafNormCos_64_16",
)

report : dict = evaluation.evaluateMoiraiMoERafCosSoftMax(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="fredMd",
    collection="moiraiMoERafNormCos_128_16",
)

report : dict = evaluation.evaluateMoiraiMoERafCosSoftMax(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="nn5",
    collection="moiraiMoERafNormCos_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERafCosSoftMax(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="nn5",
    collection="moiraiMoERafNormCos_64_16",
)

report : dict = evaluation.evaluateMoiraiMoERafCosSoftMax(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="nn5",
    collection="moiraiMoERafNormCos_128_16",
)

report : dict = evaluation.compileReports(reportOriginName="evaluationReportsMoiraiMoERafCosSoftMax",reportTargetName="evaluationFinalReportMoiraiMoERafCosSoftMax")
