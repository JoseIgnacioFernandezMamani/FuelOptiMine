from typing import Optional
from .core import RecoverableExtractionError


class UnsupportedFormatError(RecoverableExtractionError):
    """Unsupported file format"""

    def __init__(
        self,
        file_path: str,
        format_type: str,
        supported_formats: list,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(
            message=f"Unsupported format ({format_type}). Valid formats: {supported_formats}",
            cause=cause,
            file_path=file_path,
        )
