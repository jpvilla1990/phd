# PhD


## Examples

### Inference

#### MoiraiMoE

```python
import pandas as pd
from gluonts.model.forecast import SampleForecast
from datasetsModule.datasets import Datasets
from model.moiraiMoe import MoiraiMoE

CONTEXT : int = 64
PREDICTION : int = 16
NUMBER_SAMPLES : int = 100
DATASET : str = "ET"
SUBDATASET : str = "ETTh1"

dataset : Datasets = Datasets()
model : MoiraiMoE = MoiraiMoE(
    predictionLength = PREDICTION,
    contextLength = CONTEXT,
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

#### MoiraiMoE RAG
```python
from evaluation.evaluation import Evaluation

CONTEXT : int = 32
PREDICTION : int = 16
NUMBER_SAMPLES : int = 100
DATASET : str = "solarEnergy"
COLLECTION : str = "moiraiMoESolarPowerCosine_128_16"

evaluation : Evaluation = Evaluation()

report : dict = evaluation.evaluateMoiraiMoERag(
    CONTEXT,
    PREDICTION,
    NUMBER_SAMPLES,
    DATASET,
    COLLECTION,
)

print(report)
```

#### ChatTime
```python
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
```

After performing the evaluation for each desired scenario and dataset, the final evaluation will compile all the results in a PDF file

#### Final Evaluation
```python
from evaluation.evaluation import Evaluation

evaluation : Evaluation = Evaluation()

report : dict = evaluation.compileReports(reportOriginName="evaluationReportsMoiraiMoE",reportTargetName="evaluationFinalReport") # Reports will be located in src/data/reports.pdf
```

#### Ingest datasets to MoiraiMoE collections
```python
from vectorDB.vectorDBingestion import VectorDBIngestion

vectorDBingestion : VectorDBIngestion = VectorDBIngestion()

vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoECosine_32_16")
```

#### Inference RAG MoiraiMoE
```python
import pandas as pd
import numpy as np
from datasetsModule.datasets import Datasets
from model.moiraiMoe import MoiraiMoE
from utils.utils import Utils

CONTEXT : int = 128
PREDICTION : int = 16
NUMBER_SAMPLES : int = 100
DATASET : str = "covid19Deaths"
SUBDATASET : str = "covid19Deaths"
COLLECTION : str = "moiraiMoECosine_128_16"

from vectorDB.vectorDBingestion import VectorDBIngestion

vectorDBingestion : VectorDBIngestion = VectorDBIngestion()
vectorDBingestion.ingestDatasetsMoiraiMoE("moiraiMoECosine_128_16")

dataset : Datasets = Datasets()
model : MoiraiMoE = MoiraiMoE(
    predictionLength = PREDICTION,
    contextLength = CONTEXT,
    numSamples = NUMBER_SAMPLES,
)

model.setRagCollection(COLLECTION, DATASET)

iterator = dataset.loadDataset(DATASET)
features = list(iterator.getAvailableFeatures(SUBDATASET).keys())[0:2] # To effectively predict the model requires the first column is the timestamp
iterator.setSampleSize(CONTEXT + PREDICTION)
iterator.resetIteration(SUBDATASET, True)
while True:
    sample : pd.core.frame.DataFrame = iterator.iterateDataset(SUBDATASET, features)
    sample.columns = ["datetime", "value"]
    #pred : np.ndarray = model.inference(sample, DATASET)
    pred : np.ndarray = model.ragInference(sample.iloc[:CONTEXT], DATASET, True, True, plot=True)
    Utils.plot(
        [
            sample["value"].tolist(),
        ],
        "sample.png",
        "-",
        CONTEXT,
    )
    exit()
    if sample is None: # Iterate all samples until the iterator returns empty
        break
```

#### Test vector database ingestion and query
```python
import pandas as pd
import numpy as np
from gluonts.model.forecast import SampleForecast
from datasetsModule.datasets import Datasets
from model.moiraiMoe import MoiraiMoE

CONTEXT : int = 128
PREDICTION : int = 16
NUMBER_SAMPLES : int = 100
DATASET : str = "ET"
SUBDATASET : str = "ETTh1"

dataset : Datasets = Datasets()
model : MoiraiMoE = MoiraiMoE(
    predictionLength = PREDICTION,
    contextLenght = CONTEXT,
    numSamples = NUMBER_SAMPLES,
    collectionName="moiraiMoEAllCosine",
)

iterator = dataset.loadDataset(DATASET)
features = list(iterator.getAvailableFeatures(SUBDATASET).keys())[0:2] # To effectively predict the model requires the first column is the timestamp
iterator.setSampleSize(CONTEXT + PREDICTION)
iterator.resetIteration(SUBDATASET, True)
sample : pd.core.frame.DataFrame = iterator.iterateDataset(SUBDATASET, features)
sampleCopy = sample.copy()
model.ingestVector(sample[1].iloc[:CONTEXT].values, sample[1].iloc[CONTEXT:CONTEXT+PREDICTION].values)
queried : np.ndarray = model.queryVector(sampleCopy[1].iloc[:CONTEXT].values, 1)

print("original")
print(sample[1].values)
print("queried")
print(queried)
```