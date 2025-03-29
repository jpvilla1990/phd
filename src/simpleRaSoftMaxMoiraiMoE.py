from vectorDB.vectorDBingestion import VectorDBIngestion
from evaluation.evaluation import Evaluation

vectorDBingestion : VectorDBIngestion = VectorDBIngestion()
evaluation : Evaluation = Evaluation()

vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoECosine_520_30")
vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoECosine_520_96")
vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoECosine_520_336")

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=520,
    predictionLength=30,
    numberSamples=100,
    dataset="ET",
    collection="moiraiMoECosine_520_30",
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=520,
    predictionLength=96,
    numberSamples=100,
    dataset="ET",
    collection="moiraiMoECosine_520_96",
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=520,
    predictionLength=336,
    numberSamples=100,
    dataset="ET",
    collection="moiraiMoECosine_520_336",
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=520,
    predictionLength=30,
    numberSamples=100,
    dataset="power",
    collection="moiraiMoECosine_520_30",
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=520,
    predictionLength=96,
    numberSamples=100,
    dataset="power",
    collection="moiraiMoECosine_520_96",
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=520,
    predictionLength=336,
    numberSamples=100,
    dataset="power",
    collection="moiraiMoECosine_520_336",
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=520,
    predictionLength=30,
    numberSamples=100,
    dataset="m4-monthly",
    collection="moiraiMoECosine_520_30",
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=520,
    predictionLength=96,
    numberSamples=100,
    dataset="m4-monthly",
    collection="moiraiMoECosine_520_96",
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=520,
    predictionLength=96,
    numberSamples=100,
    dataset="m4-monthly",
    collection="moiraiMoECosine_520_336",
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=520,
    predictionLength=30,
    numberSamples=100,
    dataset="traffic",
    collection="moiraiMoECosine_520_30",
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=520,
    predictionLength=96,
    numberSamples=100,
    dataset="traffic",
    collection="moiraiMoECosine_520_96",
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=520,
    predictionLength=336,
    numberSamples=100,
    dataset="traffic",
    collection="moiraiMoECosine_520_336",
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=520,
    predictionLength=30,
    numberSamples=100,
    dataset="huaweiCloud",
    collection="moiraiMoECosine_520_30",
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=520,
    predictionLength=96,
    numberSamples=100,
    dataset="huaweiCloud",
    collection="moiraiMoECosine_520_96",
)

report : dict = evaluation.evaluateMoiraiMoERagSoftMax(
    contextLength=520,
    predictionLength=336,
    numberSamples=100,
    dataset="huaweiCloud",
    collection="moiraiMoECosine_520_336",
)

report : dict = evaluation.compileReports(reportOriginName="evaluationReportsMoiraiMoERagSoftMax",reportTargetName="evaluationFinalReportMoiraiMoERagSoftMax")