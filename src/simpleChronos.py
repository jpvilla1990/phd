from vectorDB.vectorDBingestion import VectorDBIngestion
from evaluation.evaluation import Evaluation
from trainingModule.training import Training

evaluation : Evaluation = Evaluation()
report : dict = evaluation.evaluateChronos(
    contextLength=32,
    predictionLength=16,
    dataset="ET",
)

report : dict = evaluation.evaluateChronos(
    contextLength=32,
    predictionLength=16,
    dataset="power",
)

report : dict = evaluation.evaluateChronos(
    contextLength=32,
    predictionLength=16,
    dataset="traffic",
)

report : dict = evaluation.evaluateChronos(
    contextLength=32,
    predictionLength=16,
    dataset="huaweiCloud",
)

report : dict = evaluation.evaluateChronos(
    contextLength=64,
    predictionLength=16,
    dataset="ET",
)

report : dict = evaluation.evaluateChronos(
    contextLength=64,
    predictionLength=16,
    dataset="power",
)

report : dict = evaluation.evaluateChronos(
    contextLength=64,
    predictionLength=16,
    dataset="traffic",
)

report : dict = evaluation.evaluateChronos(
    contextLength=64,
    predictionLength=16,
    dataset="huaweiCloud",
)

report : dict = evaluation.evaluateChronos(
    contextLength=128,
    predictionLength=16,
    dataset="ET",
)

report : dict = evaluation.evaluateChronos(
    contextLength=128,
    predictionLength=16,
    dataset="power",
)

report : dict = evaluation.evaluateChronos(
    contextLength=128,
    predictionLength=16,
    dataset="traffic",
)

report : dict = evaluation.evaluateChronos(
    contextLength=128,
    predictionLength=16,
    dataset="huaweiCloud",
)
