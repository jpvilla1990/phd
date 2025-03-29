from evaluation.evaluation import Evaluation

evaluation : Evaluation = Evaluation()

report : dict = evaluation.evaluateLinearModel(
    contextLength=520,
    predictionLength=30,
    dataset="ET",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=520,
    predictionLength=96,
    dataset="ET",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=520,
    predictionLength=336,
    dataset="ET",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=520,
    predictionLength=30,
    dataset="power",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=520,
    predictionLength=96,
    dataset="power",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=520,
    predictionLength=336,
    dataset="power",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=520,
    predictionLength=30,
    dataset="m4-monthly",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=520,
    predictionLength=96,
    dataset="m4-monthly",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=520,
    predictionLength=336,
    dataset="m4-monthly",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=520,
    predictionLength=30,
    dataset="traffic",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=520,
    predictionLength=96,
    dataset="traffic",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=520,
    predictionLength=336,
    dataset="traffic",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=520,
    predictionLength=30,
    dataset="huaweiCloud",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=520,
    predictionLength=96,
    dataset="huaweiCloud",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=520,
    predictionLength=336,
    dataset="huaweiCloud",
)

report : dict = evaluation.compileReports(reportOriginName="evaluationReportsLineal",reportTargetName="evaluationFinalReportLineal")