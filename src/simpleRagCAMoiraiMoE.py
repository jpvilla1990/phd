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
#    contextLength=128,
#    predictionLength=16,
#    numberSamples=100,
#    dataset="lotsaData",
#    trainSet=True,
#)
#report : dict = evaluation.compileReports(reportOriginName="evaluationReportsMoiraiMoE",reportTargetName="evaluationFinalReportMoiraiMoE")
training.saveModelState("RagCA-lotsaData-epoch=00-step=1485-train_loss=-1.89.ckpt")
report : dict = evaluation.evaluateMoiraiMoERagCA(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="lotsaData",
    collection="moiraiMoETrainingCosine_128_16",
    trainSet=True,
)
report : dict = evaluation.compileReports(reportOriginName="evaluationReportsMoiraiMoERagCA",reportTargetName="evaluationFinalReportMoiraiMoERagCA")
exit()
report : dict = evaluation.evaluateMoiraiMoERagCA(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="ET",
    collection="moiraiMoECosine_128_16",
)

report : dict = evaluation.evaluateMoiraiMoERagCA(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="power",
    collection="moiraiMoECosine_128_16",
)

report : dict = evaluation.evaluateMoiraiMoERagCA(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="covid19Deaths",
    collection="moiraiMoECosine_128_16",
)

report : dict = evaluation.evaluateMoiraiMoERagCA(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="electricityUCI",
    collection="moiraiMoECosine_128_16",
)

report : dict = evaluation.evaluateMoiraiMoERagCA(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="fredMd",
    collection="moiraiMoECosine_128_16",
)

report : dict = evaluation.evaluateMoiraiMoERagCA(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="nn5",
    collection="moiraiMoECosine_128_16",
)

report : dict = evaluation.evaluateMoiraiMoERagCA(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="m4-monthly",
    collection="moiraiMoECosine_128_16",
)

report : dict = evaluation.evaluateMoiraiMoERagCA(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="traffic",
    collection="moiraiMoECosine_128_16",
)

report : dict = evaluation.evaluateMoiraiMoERagCA(
    contextLength=128,
    predictionLength=16,
    numberSamples=100,
    dataset="huaweiCloud",
    collection="moiraiMoECosine_128_16",
)

report : dict = evaluation.compileReports(reportOriginName="evaluationReportsMoiraiMoERagCA",reportTargetName="evaluationFinalReportMoiraiMoERagCA")