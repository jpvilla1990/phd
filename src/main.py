from evaluation.evaluation import Evaluation

CONTEXT : int = 32
PREDICTION : int = 16
NUMBER_SAMPLES : int = 100
DATASET : str = "ET"
COLLECTION : str = "moiraiMoECosine_32_16"

evaluation : Evaluation = Evaluation()

report : dict = evaluation.evaluateMoiraiMoERag(
    CONTEXT,
    PREDICTION,
    NUMBER_SAMPLES,
    DATASET,
    COLLECTION,
)