from typing import Optional
from .core import RecoverableExtractionError

class DataLoadingWarning(RecoverableExtractionError):
    """Gets the most relevant frame from the traceback"""
    def __init__(self, file_path: str, details: str, cause: Optional[Exception] = None, **kwargs):
        super().__init__(
            message="File loading failed",
            file_path=file_path,
            details=details,
            cause=cause,  
            **kwargs
        )