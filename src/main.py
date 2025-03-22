from evaluation.evaluation import Evaluation

evaluation : Evaluation = Evaluation()

CONTEXT : int = 128
PREDICTION : int = 16
NUMBER_SAMPLES : int = 100
DATASET : str = "traffic"

report : dict = evaluation.evaluateMoiraiMoE(
    CONTEXT,
    PREDICTION,
    NUMBER_SAMPLES,
    DATASET,
)
