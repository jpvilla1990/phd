from vectorDB.vectorDBingestion import VectorDBIngestion
from evaluation.evaluation import Evaluation
from trainingModule.training import Training

vectorDBingestion : VectorDBIngestion = VectorDBIngestion()
evaluation : Evaluation = Evaluation()
training :Training = Training()

#vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoECosine_64_16")
#vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoECosine_64_16")
#vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoECosine_64_16")

#vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoEL2_64_16")
#vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoEL2_64_16")
#vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoEL2_64_16")

vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoERafL2_64_16", raf=True)
vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoERafL2_64_16", raf=True)
vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoERafL2_128_16", raf=True)

#report : dict = evaluation.evaluateMoiraiMoE(
#    contextLength=64,
#    predictionLength=16,
#    numberSamples=100,
#    dataset="lotsaData",
#    trainSet=True,
#)
#report : dict = evaluation.compileReports(reportOriginName="evaluationReportsMoiraiMoE",reportTargetName="evaluationFinalReportMoiraiMoE")
modelRagCa : str = "RagCA-2-ragCA-lotsaData-epoch=01-step=000100.ckpt"
print(modelRagCa)
loadPretrainedModel : bool = False
if loadPretrainedModel:
    training.saveModelState(modelRagCa)
#report : dict = evaluation.evaluateMoiraiMoERagCA(
#    contextLength=64,
#    predictionLength=16,
#    numberSamples=100,
#    dataset="lotsaData",
#    collection="moiraiMoETrainingCosine_64_16",
#    trainSet=True,
#)
#report : dict = evaluation.evaluateMoiraiMoERagCA(
#    contextLength=64,
#    predictionLength=16,
#    numberSamples=100,
#    dataset="lotsaData",
#    collection="moiraiMoETrainingCosine_64_16",
#    trainSet=True,
#)
report : dict = evaluation.evaluateMoiraiMoERagCA(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="ET",
    collection="moiraiMoERafL2_64_16",
    loadPretrainedRagCA=loadPretrainedModel,
)
report : dict = evaluation.evaluateMoiraiMoERagCA(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="electricityUCI",
    collection="moiraiMoERafL2_64_16",
    loadPretrainedRagCA=loadPretrainedModel,
)

report : dict = evaluation.evaluateMoiraiMoERagCA(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="huaweiCloud",
    collection="moiraiMoERafL2_64_16",
    loadPretrainedRagCA=loadPretrainedModel,
)
report : dict = evaluation.evaluateMoiraiMoERagCA(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="power",
    collection="moiraiMoERafL2_64_16",
    loadPretrainedRagCA=loadPretrainedModel,
)

report : dict = evaluation.evaluateMoiraiMoERagCA(
    contextLength=64,
    predictionLength=16,
    numberSamples=100,
    dataset="traffic",
    collection="moiraiMoERafL2_64_16",
    loadPretrainedRagCA=loadPretrainedModel,
)