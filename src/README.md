# PhD


## Examples

### Inference

#### MoiraiMoE

```python
import pandas as pd
from gluonts.model.forecast import SampleForecast
from datasets.datasets import Datasets
from model.moiraiMoe import MoiraiMoE

CONTEXT : int = 64
PREDICTION : int = 16
NUMBER_SAMPLES : int = 100
DATASET : str = "ET"
SUBDATASET : str = "ETTh1"

dataset : Datasets = Datasets()
model : MoiraiMoE = MoiraiMoE(
    predictionLength = PREDICTION,
    contextLenght = CONTEXT,
    numSamples = NUMBER_SAMPLES,
)

iterator = dataset.loadDataset(DATASET)
features = list(iterator.getAvailableFeatures(SUBDATASET).keys())[0:2] # To effectively predict the model requires the first column is the timestamp
iterator.setSampleSize(CONTEXT + PREDICTION)
iterator.resetIteration(SUBDATASET, True)
while True:
    sample : pd.core.frame.DataFrame = iterator.iterateDataset(SUBDATASET, features)
    pred : SampleForecast = model.inference(sample, DATASET)
    model.plotSample(
        sample.iloc[:CONTEXT],
        sample.iloc[CONTEXT:CONTEXT+PREDICTION],
        DATASET,
    )
    if sample is None: # Iterate all samples until the iterator returns empty
        break
```

### Evaluation

#### Final Evaluation
```python
from evaluation.evaluation import Evaluation

evaluation : Evaluation = Evaluation()

report : dict = evaluation.compileReports() # Reports will be located in src/data/reports.pdf
```

#### MoiraiMoE
```python
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
```