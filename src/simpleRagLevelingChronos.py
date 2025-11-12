from vectorDB.vectorDBingestion import VectorDBIngestion
from evaluation.evaluation import Evaluation
from trainingModule.training import Training

vectorDBingestion : VectorDBIngestion = VectorDBIngestion()
evaluation : Evaluation = Evaluation()
training :Training = Training()

vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoECosine_32_16")
vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoECosine_64_16")
vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoECosine_128_16")

vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoEL2_32_16")
vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoEL2_64_16")
vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoEL2_128_16")

vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoERafL2_32_16", raf=True)
vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoERafL2_64_16", raf=True)
vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoERafL2_128_16", raf=True)

bolt : bool = True

#report : dict = evaluation.compileReports(reportOriginName="evaluationReportsMoiraiMoERagCA",reportTargetName="evaluationFinalReportMoiraiMoERagCA")
report : dict = evaluation.evaluateChronosRagLeveling(
    contextLength=64,
    predictionLength=16,
    dataset="ET",
    collection="moiraiMoERafL2_64_16",
    bolt=bolt,
)

report : dict = evaluation.evaluateChronosRagLeveling(
    contextLength=64,
    predictionLength=16,
    dataset="power",
    collection="moiraiMoERafL2_64_16",
    bolt=bolt,
)

report : dict = evaluation.evaluateChronosRagLeveling(
    contextLength=64,
    predictionLength=16,
    dataset="traffic",
    collection="moiraiMoERafL2_64_16",
    bolt=bolt,
)

report : dict = evaluation.evaluateChronosRagLeveling(
    contextLength=64,
    predictionLength=16,
    dataset="huaweiCloud",
    collection="moiraiMoERafL2_64_16",
    bolt=bolt,
)
