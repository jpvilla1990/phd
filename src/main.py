from evaluation.evaluation import Evaluation

CONTEXT : int = 64
PREDICTION : int = 16
NUMBER_SAMPLES : int = 100
DATASET : str = "ET"
COLLECTION : str = "moiraiMoECosine_64_16"

evaluation : Evaluation = Evaluation()

report : dict = evaluation.evaluateMoiraiMoERag(
    CONTEXT,
    PREDICTION,
    NUMBER_SAMPLES,
    DATASET,
    COLLECTION,
)

from evaluation.evaluation import Evaluation

CONTEXT : int = 128
PREDICTION : int = 16
NUMBER_SAMPLES : int = 100
DATASET : str = "ET"
COLLECTION : str = "moiraiMoECosine_128_16"

evaluation : Evaluation = Evaluation()

report : dict = evaluation.evaluateMoiraiMoERag(
    CONTEXT,
    PREDICTION,
    NUMBER_SAMPLES,
    DATASET,
    COLLECTION,
)

from evaluation.evaluation import Evaluation

CONTEXT : int = 32
PREDICTION : int = 16
NUMBER_SAMPLES : int = 100
DATASET : str = "power"
COLLECTION : str = "moiraiMoECosine_32_16"

evaluation : Evaluation = Evaluation()

report : dict = evaluation.evaluateMoiraiMoERag(
    CONTEXT,
    PREDICTION,
    NUMBER_SAMPLES,
    DATASET,
    COLLECTION,
)

from evaluation.evaluation import Evaluation

CONTEXT : int = 64
PREDICTION : int = 16
NUMBER_SAMPLES : int = 100
DATASET : str = "power"
COLLECTION : str = "moiraiMoECosine_64_16"

evaluation : Evaluation = Evaluation()

report : dict = evaluation.evaluateMoiraiMoERag(
    CONTEXT,
    PREDICTION,
    NUMBER_SAMPLES,
    DATASET,
    COLLECTION,
)

from evaluation.evaluation import Evaluation

CONTEXT : int = 128
PREDICTION : int = 16
NUMBER_SAMPLES : int = 100
DATASET : str = "power"
COLLECTION : str = "moiraiMoECosine_128_16"

evaluation : Evaluation = Evaluation()

report : dict = evaluation.evaluateMoiraiMoERag(
    CONTEXT,
    PREDICTION,
    NUMBER_SAMPLES,
    DATASET,
    COLLECTION,
)

from evaluation.evaluation import Evaluation

CONTEXT : int = 32
PREDICTION : int = 16
NUMBER_SAMPLES : int = 100
DATASET : str = "solarEnergy"
COLLECTION : str = "moiraiMoECosine_32_16"

evaluation : Evaluation = Evaluation()

report : dict = evaluation.evaluateMoiraiMoERag(
    CONTEXT,
    PREDICTION,
    NUMBER_SAMPLES,
    DATASET,
    COLLECTION,
)