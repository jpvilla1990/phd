import pandas as pd
import torch
from model.chronosModel import Chronos

model : Chronos = Chronos()

df = pd.read_csv(
    "https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv"
)

# context must be either a 1D tensor, a list of 1D tensors,
# or a left-padded 2D tensor with batch as the first dimension
# quantiles is an fp32 tensor with shape [batch_size, prediction_length, num_quantile_levels]
# mean is an fp32 tensor with shape [batch_size, prediction_length]
prediction = model.predict(
    sample=torch.tensor(df["#Passengers"]),
    predictionLength=12,
)

print(prediction.shape)