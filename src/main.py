from evaluation.evaluation import Evaluation

evaluation : Evaluation = Evaluation()

report : dict = evaluation.compileReports(reportOriginName="evaluationReportsMoiraiMoERag",reportTargetName="evaluationFinalReportRag") # Reports will be located in src/data/reports.pdf