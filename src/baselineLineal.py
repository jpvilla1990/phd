from evaluation.evaluation import Evaluation

evaluation : Evaluation = Evaluation()

report : dict = evaluation.evaluateLinearModel(
    contextLength=32,
    predictionLength=16,
    dataset="ET",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=64,
    predictionLength=16,
    dataset="ET",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=128,
    predictionLength=16,
    dataset="ET",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=32,
    predictionLength=16,
    dataset="power",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=64,
    predictionLength=16,
    dataset="power",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=128,
    predictionLength=16,
    dataset="power",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=32,
    predictionLength=16,
    dataset="m4-monthly",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=64,
    predictionLength=16,
    dataset="m4-monthly",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=128,
    predictionLength=16,
    dataset="m4-monthly",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=32,
    predictionLength=16,
    dataset="traffic",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=64,
    predictionLength=16,
    dataset="traffic",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=128,
    predictionLength=16,
    dataset="traffic",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=32,
    predictionLength=16,
    dataset="huaweiCloud",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=64,
    predictionLength=16,
    dataset="huaweiCloud",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=128,
    predictionLength=16,
    dataset="huaweiCloud",
)

report : dict = evaluation.compileReports(reportOriginName="evaluationReportsLineal",reportTargetName="evaluationFinalReportLineal")