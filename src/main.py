from evaluation.evaluation import Evaluation

evaluation : Evaluation = Evaluation()

report : dict = evaluation.compileReports(reportOriginName="evaluationReportsMoiraiMoERag",reportTargetName="evaluationFinalReportMoiraiMoERag") # Reports will be located in src/data/reports.pdf

report : dict = evaluation.compileReports(reportOriginName="evaluationReportsChatTime",reportTargetName="evaluationFinalReportChatTime") # Reports will be located in src/data/reports.pdf
