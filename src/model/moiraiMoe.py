import torch
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from huggingface_hub import hf_hub_download

import matplotlib.pyplot as plt
from uni2ts.eval_util.plot import plot_single

from gluonts.torch import PyTorchPredictor
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule
from gluonts.dataset.common import ListDataset
from gluonts.model.forecast import SampleForecast

from exceptions.modelException import ModelException

class MoiraiMoE(object):
    """
    Class to handle MoiraiMoe
    """
    def __init__(
        self,
        modelSize : str = "small",
        predictionLength : int = 20,
        contextLenght : int = 200,
        patchSize : int = 16,
        numSamples : int = 100,
        targetDim : int = 1,
        featDynamicRealDim : int = 0,
        pastFeatDynamicRealDim : int = 0,
        batchSize : int = 1,
        freq : str = "H",
    ):
        self.__freq : str = freq
        self.__contextLenght : int = contextLenght
        self.__model : MoiraiMoEForecast = MoiraiMoEForecast(
            module=MoiraiMoEModule.from_pretrained(f"Salesforce/moirai-moe-1.0-R-{modelSize}"),
            prediction_length=predictionLength,
            context_length=contextLenght,
            patch_size=patchSize,
            num_samples=numSamples,
            target_dim=targetDim,
            feat_dynamic_real_dim=featDynamicRealDim,
            past_feat_dynamic_real_dim=pastFeatDynamicRealDim,
        )

        self.__predictor : PyTorchPredictor = self.__model.create_predictor(batch_size=batchSize)

    def inference(self, sample : pd.core.frame.DataFrame) -> SampleForecast:
        """
        Method to predict one sample
        """
        if len(sample.columns) != 2:
            raise ModelException("MoiraiMoE predictor accepts only two columns, timestamp and timeseries itself")
        sample.columns = ["datetime", "value"]
        sampleGluonts : ListDataset = ListDataset(
            [{"start": sample["datetime"].iloc[0], "target": sample["value"].tolist()}],
            freq=self.__freq  # Set frequency to hourly
        )
        return next(iter(self.__predictor.predict(sampleGluonts)))
    
    def plotSample(self, sample : pd.core.frame.DataFrame, groundTruth : pd.core.frame.DataFrame):
        """
        Method to plot sample
        """
        if len(sample.columns) != 2 or len(groundTruth.columns) != 2:
            raise ModelException("MoiraiMoE predictor accepts only two columns, timestamp and timeseries itself")
        sample.columns = ["datetime", "value"]
        groundTruth.columns = ["datetime", "value"]

        sampleDict : dict = {"start": sample["datetime"].iloc[0], "target": sample["value"].tolist()}
        groundTruthDict : dict = {"start": groundTruth["datetime"].iloc[0], "target": groundTruth["value"].tolist()}
        sampleGluonts : ListDataset = ListDataset(
            [sampleDict],
            freq=self.__freq  # Set frequency to hourly
        )

        prediction : SampleForecast = next(iter(self.__predictor.predict(sampleGluonts)))

        plot_single(
            sampleDict,
            groundTruthDict,
            prediction,
            context_length=self.__contextLenght,
            name="pred",
            show_label=True,
        )
        plt.show()


#predictor = model.createPredictor(batchSize=32)
#forecasts = predictor.predict(test_data.input)

#input_it = iter(test_data.input)
#label_it = iter(test_data.label)
#forecast_it = iter(forecasts)

#inp = next(input_it)
#label = next(label_it)
#forecast = next(forecast_it)

#print(inp)
#print(label)
#print(forecast)

#plot_single(
#    inp, 
#    label, 
#    forecast, 
#    context_length=200,
#    name="pred",
#    show_label=True,
#)
#plt.show()