import chromadb
from utils.fileSystem import FileSystem

class VectorDB(FileSystem):
    """
    Class to handle vector DB
    """
    def __init__(self):
        super().__init__()