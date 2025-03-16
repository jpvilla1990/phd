import os
import yaml
import pyarrow
import pyarrow.feather as feather
import pandas as pd

class Utils(object):
    """
    Class to store util functions
    """
    def appendPath(rootPath : str, paths : list) -> str:
        """
        Static method to recurrently append paths in a long path
        """
        rootPath = os.path.join(rootPath, paths[0])
        if len(paths) == 1:
            return rootPath
        else:
            return Utils.appendPath(rootPath, paths[1:])
        
    def readYaml(filePath : str) -> dict:
        """
        Static method to read from a yaml file
        """
        content : dict
        with open(filePath, "r") as file:
            content = yaml.safe_load(file)

        return content

    def writeYaml(filePath : str, content : dict):
        """
        Static method to write in a yaml file
        """
        with open(filePath, "w") as file:
            yaml.safe_dump(content, file)

    def savePandasAsArrow(dataframe : pd.core.frame.DataFrame, path : str):
        """
        Static method to save pandas dataframe as arrow
        """
        table : pyarrow.lib.Table = pyarrow.Table.from_pandas(dataframe)
        feather.write_feather(table, f"{path}.arrow")

    def loadPandasFromArrow(path : str) -> pd.core.frame.DataFrame:
        """
        Method to load pandas from arrow file
        """
        table : pyarrow.lib.Table = feather.read_table(f"{path}.arrow")
        return table.to_pandas()