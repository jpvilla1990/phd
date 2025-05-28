from vectorDB.vectorDBingestion import VectorDBIngestion
from evaluation.evaluation import Evaluation
from trainingModule.training import Training

vectorDBingestion : VectorDBIngestion = VectorDBIngestion()
evaluation : Evaluation = Evaluation()
training :Training = Training()

vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoECosine_32_16")
vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoECosine_64_16")
vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoECosine_128_16")

#report : dict = evaluation.evaluateMoiraiMoE(
#    contextLength=32,
#    predictionLength=16,
#    numberSamples=100,
#    dataset="lotsaData",
#    trainSet=True,
#)
#report : dict = evaluation.compileReports(reportOriginName="evaluationReportsMoiraiMoE",reportTargetName="evaluationFinalReportMoiraiMoE")
training.saveModelState("RagCA-128_16-lotsaData-epoch=40-step=040300.ckpt")
#report : dict = evaluation.evaluateMoiraiMoERagCA(
#    contextLength=32,
#    predictionLength=16,
#    numberSamples=100,
#    dataset="lotsaData",
#    collection="moiraiMoETrainingCosine_128_16",
#    trainSet=True,
#)
#report : dict = evaluation.evaluateMoiraiMoERagCA(
#    contextLength=32,
#    predictionLength=16,
#    numberSamples=100,
#    dataset="lotsaData",
#    collection="moiraiMoETrainingCosine_32_16",
#    trainSet=True,
#)
#report : dict = evaluation.evaluateMoiraiMoERagCA(
#    contextLength=32,
#    predictionLength=16,
#    numberSamples=100,
#    dataset="lotsaData",
#    collection="moiraiMoETrainingCosine_32_16",
#    trainSet=True,
#)
#report : dict = evaluation.compileReports(reportOriginName="evaluationReportsMoiraiMoERagCA",reportTargetName="evaluationFinalReportMoiraiMoERagCA")
report : dict = evaluation.evaluateMoiraiMoERagCA(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="ET",
    collection="moiraiMoECosine_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERagCA(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="power",
    collection="moiraiMoECosine_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERagCA(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="covid19Deaths",
    collection="moiraiMoECosine_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERagCA(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="electricityUCI",
    collection="moiraiMoECosine_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERagCA(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="fredMd",
    collection="moiraiMoECosine_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERagCA(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="nn5",
    collection="moiraiMoECosine_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERagCA(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="m4-monthly",
    collection="moiraiMoECosine_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERagCA(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="traffic",
    collection="moiraiMoECosine_32_16",
)

report : dict = evaluation.evaluateMoiraiMoERagCA(
    contextLength=32,
    predictionLength=16,
    numberSamples=100,
    dataset="huaweiCloud",
    collection="moiraiMoECosine_32_16",
)

report : dict = evaluation.compileReports(reportOriginName="evaluationReportsMoiraiMoERagCA",reportTargetName="evaluationFinalReportMoiraiMoERagCA")