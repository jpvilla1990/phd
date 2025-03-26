from evaluation.evaluation import Evaluation

evaluation : Evaluation = Evaluation()

report : dict = evaluation.evaluateChatTimes(
    contextLength=32,
    predictionLength=16,
    dataset="ET",
)

report : dict = evaluation.evaluateChatTimes(
    contextLength=64,
    predictionLength=16,
    dataset="ET",
)

report : dict = evaluation.evaluateChatTimes(
    contextLength=128,
    predictionLength=16,
    dataset="ET",
)

report : dict = evaluation.evaluateChatTimes(
    contextLength=32,
    predictionLength=16,
    dataset="solarEnergy",
)

report : dict = evaluation.evaluateChatTimes(
    contextLength=32,
    predictionLength=16,
    dataset="power",
)

report : dict = evaluation.evaluateChatTimes(
    contextLength=64,
    predictionLength=16,
    dataset="power",
)

report : dict = evaluation.evaluateChatTimes(
    contextLength=128,
    predictionLength=16,
    dataset="power",
)

report : dict = evaluation.evaluateChatTimes(
    contextLength=32,
    predictionLength=16,
    dataset="m4-monthly",
)

report : dict = evaluation.evaluateChatTimes(
    contextLength=64,
    predictionLength=16,
    dataset="m4-monthly",
)

report : dict = evaluation.evaluateChatTimes(
    contextLength=128,
    predictionLength=16,
    dataset="m4-monthly",
)

report : dict = evaluation.evaluateChatTimes(
    contextLength=32,
    predictionLength=16,
    dataset="traffic",
)

report : dict = evaluation.evaluateChatTimes(
    contextLength=64,
    predictionLength=16,
    dataset="traffic",
)

report : dict = evaluation.evaluateChatTimes(
    contextLength=128,
    predictionLength=16,
    dataset="traffic",
)

report : dict = evaluation.evaluateChatTimes(
    contextLength=32,
    predictionLength=16,
    dataset="huaweiCloud",
)

report : dict = evaluation.evaluateChatTimes(
    contextLength=64,
    predictionLength=16,
    dataset="huaweiCloud",
)

report : dict = evaluation.evaluateChatTimes(
    contextLength=128,
    predictionLength=16,
    dataset="huaweiCloud",
)

report : dict = evaluation.compileReports(reportOriginName="evaluationReportsChatTime",reportTargetName="evaluationFinalReportChatTime")