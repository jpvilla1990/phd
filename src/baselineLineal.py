from evaluation.evaluation import Evaluation

evaluation : Evaluation = Evaluation()
a = """
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

report : dict = evaluation.evaluateLinearModel(
    contextLength=520,
    predictionLength=30,
    dataset="electricityUCI",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=520,
    predictionLength=96,
    dataset="electricityUCI",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=520,
    predictionLength=336,
    dataset="electricityUCI",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=128,
    predictionLength=16,
    dataset="electricityUCI",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=64,
    predictionLength=16,
    dataset="electricityUCI",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=32,
    predictionLength=16,
    dataset="electricityUCI",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=128,
    predictionLength=16,
    dataset="covid19Deaths",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=64,
    predictionLength=16,
    dataset="covid19Deaths",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=32,
    predictionLength=16,
    dataset="covid19Deaths",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=128,
    predictionLength=16,
    dataset="fredMd",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=64,
    predictionLength=16,
    dataset="fredMd",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=32,
    predictionLength=16,
    dataset="fredMd",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=520,
    predictionLength=30,
    dataset="fredMd",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=520,
    predictionLength=96,
    dataset="fredMd",
)
"""
report : dict = evaluation.evaluateLinearModel(
    contextLength=128,
    predictionLength=16,
    dataset="ET",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=128,
    predictionLength=16,
    dataset="covid19Deaths",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=128,
    predictionLength=16,
    dataset="electricityUCI",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=128,
    predictionLength=16,
    dataset="fredMd",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=128,
    predictionLength=16,
    dataset="huaweiCloud",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=128,
    predictionLength=16,
    dataset="m4-monthly",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=128,
    predictionLength=16,
    dataset="nn5",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=128,
    predictionLength=16,
    dataset="power",
)

report : dict = evaluation.evaluateLinearModel(
    contextLength=128,
    predictionLength=16,
    dataset="traffic",
)

report : dict = evaluation.compileReports(reportOriginName="evaluationReportsLineal",reportTargetName="evaluationFinalReportLineal")