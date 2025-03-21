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

CONTEXT : int = 64
PREDICTION : int = 16
DATASET : str = "ET"

evaluation : Evaluation = Evaluation()

report : dict = evaluation.evaluateChatTimes(
    CONTEXT,
    PREDICTION,
    DATASET,
)

CONTEXT : int = 128
PREDICTION : int = 16
DATASET : str = "ET"

evaluation : Evaluation = Evaluation()

report : dict = evaluation.evaluateChatTimes(
    CONTEXT,
    PREDICTION,
    DATASET,
)