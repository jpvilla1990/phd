from evaluation.evaluation import Evaluation

evaluation : Evaluation = Evaluation()

report : dict = evaluation.compileReports(reportOriginName="evaluationReportsMoiraiMoE",reportTargetName="evaluationFinalReportMoiraiMoE")

report : dict = evaluation.compileReports(reportOriginName="evaluationReportsMoiraiMoERagMean",reportTargetName="evaluationFinalReportMoiraiMoERagMean")

report : dict = evaluation.compileReports(reportOriginName="evaluationReportsMoiraiMoERagSoftMax",reportTargetName="evaluationFinalReportMoiraiMoERagSoftMax")

report : dict = evaluation.compileReports(reportOriginName="evaluationReportsChatTime",reportTargetName="evaluationFinalReportChatTime")

report : dict = evaluation.compileReports(reportOriginName="evaluationReportsChatTimeRag",reportTargetName="evaluationFinalReportChatTimeRag")

from datasetsModule.datasets import Datasets

DATASET : str = "lotsaData"

dataset : Datasets = Datasets()

iterator = dataset.loadDataset(DATASET)