from vectorDB.vectorDBingestion import VectorDBIngestion
from trainingModule.trainingRag import TrainingRag

training : TrainingRag = TrainingRag(
    collection="moiraiMoERafL2",
    dataset="ET",
    predictionLength=16,
    contextLength=32,
    numberSamples=100,
)
training.train()

training : TrainingRag = TrainingRag(
    collection="moiraiMoERafL2",
    dataset="ET",
    predictionLength=16,
    contextLength=64,
    numberSamples=100,
)
training.train()

training : TrainingRag = TrainingRag(
    collection="moiraiMoERafL2",
    dataset="ET",
    predictionLength=16,
    contextLength=128,
    numberSamples=100,
)
training.train()

training : TrainingRag = TrainingRag(
    collection="moiraiMoERafL2",
    dataset="electricityUCI",
    predictionLength=16,
    contextLength=32,
    numberSamples=100,
)
training.train()

training : TrainingRag = TrainingRag(
    collection="moiraiMoERafL2",
    dataset="electricityUCI",
    predictionLength=16,
    contextLength=64,
    numberSamples=100,
)
training.train()

training : TrainingRag = TrainingRag(
    collection="moiraiMoERafL2",
    dataset="electricityUCI",
    predictionLength=16,
    contextLength=128,
    numberSamples=100,
)
training.train()

training : TrainingRag = TrainingRag(
    collection="moiraiMoERafL2",
    dataset="huaweiCloud",
    predictionLength=16,
    contextLength=32,
    numberSamples=100,
)
training.train()

training : TrainingRag = TrainingRag(
    collection="moiraiMoERafL2",
    dataset="huaweiCloud",
    predictionLength=16,
    contextLength=64,
    numberSamples=100,
)
training.train()

training : TrainingRag = TrainingRag(
    collection="moiraiMoERafL2",
    dataset="huaweiCloud",
    predictionLength=16,
    contextLength=128,
    numberSamples=100,
)
training.train()

training : TrainingRag = TrainingRag(
    collection="moiraiMoERafL2",
    dataset="power",
    predictionLength=16,
    contextLength=32,
    numberSamples=100,
)
training.train()

training : TrainingRag = TrainingRag(
    collection="moiraiMoERafL2",
    dataset="power",
    predictionLength=16,
    contextLength=64,
    numberSamples=100,
)
training.train()

training : TrainingRag = TrainingRag(
    collection="moiraiMoERafL2",
    dataset="power",
    predictionLength=16,
    contextLength=128,
    numberSamples=100,
)
training.train()

training : TrainingRag = TrainingRag(
    collection="moiraiMoERafL2",
    dataset="traffic",
    predictionLength=16,
    contextLength=32,
    numberSamples=100,
)
training.train()

training : TrainingRag = TrainingRag(
    collection="moiraiMoERafL2",
    dataset="traffic",
    predictionLength=16,
    contextLength=64,
    numberSamples=100,
)
training.train()

training : TrainingRag = TrainingRag(
    collection="moiraiMoERafL2",
    dataset="traffic",
    predictionLength=16,
    contextLength=128,
    numberSamples=100,
)
training.train()
