from evaluation.evaluation import Evaluation

CONTEXT : int = 32
PREDICTION : int = 16
DATASET : str = "ET"

evaluation : Evaluation = Evaluation()

report : dict = evaluation.evaluateChatTimes(
    CONTEXT,
    PREDICTION,
    DATASET,
)

print(report)