from vectorDB.vectorDBingestion import VectorDBIngestion
from evaluation.evaluation import Evaluation

vectorDBingestion : VectorDBIngestion = VectorDBIngestion()
evaluation : Evaluation = Evaluation()

vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoEL2_128_16")
vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoEL2_64_16")
vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoEL2_32_16")
vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoEL2_520_30")
vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoEL2_520_96")
vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoEL2_520_336")

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="ET",
    collection="moiraiMoEL2_32_16",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="ET",
    collection="moiraiMoEL2_64_16",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="ET",
    collection="moiraiMoEL2_128_16",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=520,
    predictionLength=30,
    numberSamples=100,
    dataset="ET",
    collection="moiraiMoEL2_520_30",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=520,
    predictionLength=96,
    numberSamples=100,
    dataset="ET",
    collection="moiraiMoEL2_520_96",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=520,
    predictionLength=336,
    numberSamples=100,
    dataset="ET",
    collection="moiraiMoEL2_520_336",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="power",
    collection="moiraiMoEL2_32_16",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="power",
    collection="moiraiMoEL2_64_16",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="power",
    collection="moiraiMoEL2_128_16",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=520,
    predictionLength=30,
    numberSamples=100,
    dataset="power",
    collection="moiraiMoEL2_520_30",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=520,
    predictionLength=96,
    numberSamples=100,
    dataset="power",
    collection="moiraiMoEL2_520_96",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=520,
    predictionLength=336,
    numberSamples=100,
    dataset="power",
    collection="moiraiMoEL2_520_336",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="m4-monthly",
    collection="moiraiMoEL2_32_16",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="m4-monthly",
    collection="moiraiMoEL2_64_16",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="m4-monthly",
    collection="moiraiMoEL2_128_16",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=520,
    predictionLength=30,
    numberSamples=100,
    dataset="m4-monthly",
    collection="moiraiMoEL2_520_30",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=520,
    predictionLength=96,
    numberSamples=100,
    dataset="m4-monthly",
    collection="moiraiMoEL2_520_96",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=520,
    predictionLength=336,
    numberSamples=100,
    dataset="m4-monthly",
    collection="moiraiMoEL2_520_336",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="traffic",
    collection="moiraiMoEL2_32_16",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="traffic",
    collection="moiraiMoEL2_64_16",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="traffic",
    collection="moiraiMoEL2_128_16",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=520,
    predictionLength=30,
    numberSamples=100,
    dataset="traffic",
    collection="moiraiMoEL2_520_30",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=520,
    predictionLength=96,
    numberSamples=100,
    dataset="traffic",
    collection="moiraiMoEL2_520_96",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=520,
    predictionLength=336,
    numberSamples=100,
    dataset="traffic",
    collection="moiraiMoEL2_520_336",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="huaweiCloud",
    collection="moiraiMoEL2_32_16",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="huaweiCloud",
    collection="moiraiMoEL2_64_16",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="huaweiCloud",
    collection="moiraiMoEL2_128_16",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=520,
    predictionLength=30,
    numberSamples=100,
    dataset="huaweiCloud",
    collection="moiraiMoEL2_520_30",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=520,
    predictionLength=96,
    numberSamples=100,
    dataset="huaweiCloud",
    collection="moiraiMoEL2_520_96",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=520,
    predictionLength=336,
    numberSamples=100,
    dataset="huaweiCloud",
    collection="moiraiMoEL2_520_336",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=520,
    predictionLength=336,
    numberSamples=100,
    dataset="electricityUCI",
    collection="moiraiMoEL2_520_336",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=520,
    predictionLength=96,
    numberSamples=100,
    dataset="electricityUCI",
    collection="moiraiMoEL2_520_96",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=520,
    predictionLength=30,
    numberSamples=100,
    dataset="electricityUCI",
    collection="moiraiMoEL2_520_30",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="electricityUCI",
    collection="moiraiMoEL2_128_16",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="electricityUCI",
    collection="moiraiMoEL2_64_16",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="electricityUCI",
    collection="moiraiMoEL2_32_16",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="covid19Deaths",
    collection="moiraiMoEL2_32_16",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="covid19Deaths",
    collection="moiraiMoEL2_64_16",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="covid19Deaths",
    collection="moiraiMoEL2_128_16",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=520,
    predictionLength=96,
    numberSamples=100,
    dataset="fredMd",
    collection="moiraiMoEL2_520_96",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=520,
    predictionLength=30,
    numberSamples=100,
    dataset="fredMd",
    collection="moiraiMoEL2_520_30",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="fredMd",
    collection="moiraiMoEL2_32_16",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="fredMd",
    collection="moiraiMoEL2_64_16",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="fredMd",
    collection="moiraiMoEL2_128_16",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=520,
    predictionLength=96,
    numberSamples=100,
    dataset="nn5",
    collection="moiraiMoEL2_520_96",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=520,
    predictionLength=30,
    numberSamples=100,
    dataset="nn5",
    collection="moiraiMoEL2_520_30",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="nn5",
    collection="moiraiMoEL2_32_16",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="nn5",
    collection="moiraiMoEL2_64_16",
    cosine=False,
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="nn5",
    collection="moiraiMoEL2_128_16",
    cosine=False,
)

report : dict = evaluation.compileReports(reportOriginName="evaluationReportsMoiraiMoERagSoftMax",reportTargetName="evaluationFinalReportMoiraiMoERagL2SoftMax")