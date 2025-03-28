from vectorDB.vectorDBingestion import VectorDBIngestion
from evaluation.evaluation import Evaluation

vectorDBingestion : VectorDBIngestion = VectorDBIngestion()
evaluation : Evaluation = Evaluation()

vectorDBingestion.ingestDatasetsChatTime("chatTimeCosine_32_16")
vectorDBingestion.ingestDatasetsChatTime("chatTimeCosine_64_16")
vectorDBingestion.ingestDatasetsChatTime("chatTimeCosine_128_16")

a = """
report : dict = evaluation.evaluateChatTimesRag(
    contextLength=32,
    predictionLength=16,
    dataset="ET",
    collection="chatTimeCosine_32_16",
)
report : dict = evaluation.evaluateChatTimesRag(
    contextLength=64,
    predictionLength=16,
    dataset="ET",
    collection="chatTimeCosine_64_16",
)

report : dict = evaluation.evaluateChatTimesRag(
    contextLength=128,
    predictionLength=16,
    dataset="ET",
    collection="chatTimeCosine_128_16",
)

report : dict = evaluation.evaluateChatTimesRag(
    contextLength=32,
    predictionLength=16,
    dataset="solarEnergy",
    collection="chatTimeCosine_32_16",
)

report : dict = evaluation.evaluateChatTimesRag(
    contextLength=32,
    predictionLength=16,
    dataset="power",
    collection="chatTimeCosine_32_16",
)

report : dict = evaluation.evaluateChatTimesRag(
    contextLength=64,
    predictionLength=16,
    dataset="power",
    collection="chatTimeCosine_64_16",
)

report : dict = evaluation.evaluateChatTimesRag(
    contextLength=128,
    predictionLength=16,
    dataset="power",
    collection="chatTimeCosine_128_16",
)

report : dict = evaluation.evaluateChatTimesRag(
    contextLength=32,
    predictionLength=16,
    dataset="m4-monthly",
    collection="chatTimeCosine_32_16",
)

report : dict = evaluation.evaluateChatTimesRag(
    contextLength=64,
    predictionLength=16,
    dataset="m4-monthly",
    collection="chatTimeCosine_64_16",
)
"""
report : dict = evaluation.evaluateChatTimesRag(
    contextLength=128,
    predictionLength=16,
    dataset="m4-monthly",
    collection="chatTimeCosine_128_16",
)

report : dict = evaluation.evaluateChatTimesRag(
    contextLength=32,
    predictionLength=16,
    dataset="traffic",
    collection="chatTimeCosine_32_16",
)

report : dict = evaluation.evaluateChatTimesRag(
    contextLength=64,
    predictionLength=16,
    dataset="traffic",
    collection="chatTimeCosine_64_16",
)

report : dict = evaluation.evaluateChatTimesRag(
    contextLength=128,
    predictionLength=16,
    dataset="traffic",
    collection="chatTimeCosine_128_16",
)

report : dict = evaluation.evaluateChatTimesRag(
    contextLength=32,
    predictionLength=16,
    dataset="huaweiCloud",
    collection="chatTimeCosine_32_16",
)

report : dict = evaluation.evaluateChatTimesRag(
    contextLength=64,
    predictionLength=16,
    dataset="huaweiCloud",
    collection="chatTimeCosine_64_16",
)

report : dict = evaluation.evaluateChatTimesRag(
    contextLength=128,
    predictionLength=16,
    dataset="huaweiCloud",
    collection="chatTimeCosine_128_16",
)

report : dict = evaluation.compileReports(reportOriginName="evaluationReportsChatTimeRag",reportTargetName="evaluationFinalReportChatTimeRag")
