import numpy as np
from sklearn.linear_model import Ridge
from utils.fileSystem import FileSystem
from exceptions.modelException import ModelException

class LinealRegression(FileSystem):
    def __init__(self, contextLength : int, predictionLength : int):
        super().__init__()
        alpha : float = self._getConfig()["models"]["linealModel"]["alpha"]
        type : str = self._getConfig()["models"]["linealModel"]["type"]

        self.__refitSteps : int = self._getConfig()["models"]["linealModel"]["refitSteps"]
        self.__model : any = None

        if type == "ridge":
            self.__model = Ridge(alpha=alpha)
        else:
            raise ModelException(f"Linear Model {type} not defined")
        self.__xTrain : list = []
        self.__yTrain : list = []
        self.__step : int = 0

        self.__predictionLength : int = predictionLength
        self.__contextLength : int = contextLength

    def fit(self, X : np.ndarray, Y : np.ndarray):
        """
        Fits the model with a new training example X, Y
        """
        self.__model.fit(
            X,
            Y,
        )

    def predictWithStepFitting(self, X : np.ndarray, Y : np.ndarray) -> np.ndarray:
        """
        Method to predict and fitting every number of defined steps
        """
        if X.shape[0] != self.__contextLength or Y.shape[0] != self.__predictionLength:
            print()
            raise ModelException(f"Samples X and Y do not fit the context length {self.__contextLength} or the prediction length {self.__predictionLength}")

        if self.__step >= 2 * self.__refitSteps:
            self.__step = self.__refitSteps

        if self.__step < self.__refitSteps:
            self.__xTrain.append(X)
            self.__yTrain.append(Y)
            self.__step += 1
            return None

        elif self.__refitSteps % self.__step == 0:
            self.fit(
                np.array(self.__xTrain),
                np.array(self.__yTrain),
            )
            self.__xTrain = [X]
            self.__yTrain = [Y]
            self.__step += 1
            return self.predict(X)

        else:
            self.__xTrain.append(X)
            self.__yTrain.append(Y)
            self.__step += 1
            return self.predict(X)

    def predict(self, X : np.ndarray) -> np.ndarray:
        """
        Predicts the next horizon points using the trained model
        """
        return self.__model.predict(X.reshape(1, -1))[0]