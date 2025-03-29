from evaluation.evaluation import Evaluation

evaluation : Evaluation = Evaluation()

report : dict = evaluation.evaluateMoiraiMoE(
    contextLength=520,
    predictionLength=30,
    numberSamples=100,
    dataset="ET",
)

report : dict = evaluation.evaluateMoiraiMoE(
    contextLength=520,
    predictionLength=96,
    numberSamples=100,
    dataset="ET",
)

report : dict = evaluation.evaluateMoiraiMoE(
    contextLength=520,
    predictionLength=336,
    numberSamples=100,
    dataset="ET",
)

report : dict = evaluation.evaluateMoiraiMoE(
    contextLength=520,
    predictionLength=30,
    numberSamples=100,
    dataset="power",
)

report : dict = evaluation.evaluateMoiraiMoE(
    contextLength=520,
    predictionLength=96,
    numberSamples=100,
    dataset="power",
)

report : dict = evaluation.evaluateMoiraiMoE(
    contextLength=520,
    predictionLength=336,
    numberSamples=100,
    dataset="power",
)

report : dict = evaluation.evaluateMoiraiMoE(
    contextLength=520,
    predictionLength=30,
    numberSamples=100,
    dataset="m4-monthly",
)

report : dict = evaluation.evaluateMoiraiMoE(
    contextLength=520,
    predictionLength=96,
    numberSamples=100,
    dataset="m4-monthly",
)

report : dict = evaluation.evaluateMoiraiMoE(
    contextLength=520,
    predictionLength=336,
    numberSamples=100,
    dataset="m4-monthly",
)

report : dict = evaluation.evaluateMoiraiMoE(
    contextLength=520,
    predictionLength=30,
    numberSamples=100,
    dataset="traffic",
)

report : dict = evaluation.evaluateMoiraiMoE(
    contextLength=520,
    predictionLength=96,
    numberSamples=100,
    dataset="traffic",
)

report : dict = evaluation.evaluateMoiraiMoE(
    contextLength=520,
    predictionLength=336,
    numberSamples=100,
    dataset="traffic",
)

report : dict = evaluation.evaluateMoiraiMoE(
    contextLength=520,
    predictionLength=30,
    numberSamples=100,
    dataset="huaweiCloud",
)

report : dict = evaluation.evaluateMoiraiMoE(
    contextLength=520,
    predictionLength=96,
    numberSamples=100,
    dataset="huaweiCloud",
)

report : dict = evaluation.evaluateMoiraiMoE(
    contextLength=520,
    predictionLength=336,
    numberSamples=100,
    dataset="huaweiCloud",
)

report : dict = evaluation.compileReports(reportOriginName="evaluationReportsMoiraiMoE",reportTargetName="evaluationFinalReportMoiraiMoE")