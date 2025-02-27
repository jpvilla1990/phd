from evaluation.evaluation import Evaluation

CONTEXT : int = 32
PREDICTION : int = 16
NUMBER_SAMPLES : int = 100
DATASET : str = "m4-monthly"

evaluation : Evaluation = Evaluation()
evaluation.compileReports()

report : dict = evaluation.evaluateMoiraiMoE(
    CONTEXT,
    PREDICTION,
    NUMBER_SAMPLES,
    DATASET,
)

print(report)