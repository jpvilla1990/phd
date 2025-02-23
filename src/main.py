from evaluation.evaluation import Evaluation

CONTEXT : int = 64
PREDICTION : int = 16
NUMBER_SAMPLES : int = 100
DATASET : str = "ET"

evaluation : Evaluation = Evaluation()

report : dict = evaluation.evaluateMoiraiMoE(
    CONTEXT,
    PREDICTION,
    NUMBER_SAMPLES,
    DATASET,
)

print(report)