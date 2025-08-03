import traceback
import linecache
from pathlib import Path
from typing import Optional, Literal
from types import TracebackType, FrameType


class ExtractionError(Exception):
    """Base class for all module exceptions"""

    def __init__(
        self,
        message: str,
        dataset: Optional[str] = None,
        file_path: Optional[str] = None,
        details: Optional[str] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        self.dataset = dataset
        self.file_path = Path(file_path) if file_path else None
        self.details = details
        self.cause = cause

        full_msg: str = f"\n🚨 {message}"
        if dataset:
            full_msg += f"\n   • Dataset: {dataset}"
        if self.file_path is not None:
            full_msg += f"\n   • File: {self.file_path.name}"
        if details:
            full_msg += f"\n   • Details: {details}"

        super().__init__(full_msg)

    def __str__(self) -> str:
        """Add excecution trace to the error message"""
        original_msg: str = super().__str__()
        tb: Optional[tuple] = self._get_relevant_traceback()

        if tb:
            filename, lineno, code_line = tb
            original_msg += (
                f"\n\n   ==== ERROR SOURCE ===="
                f"\n   • File: {filename}:{lineno}"
                f"\n   • Code: {code_line.strip()}"
            )

            context: str = self._get_code_context(filename, lineno)
            if context:
                original_msg += f"\n\n   ==== CODE CONTEXT ====\n{context}"

        return original_msg

    def _get_relevant_traceback(self) -> Optional[tuple]:
        """Gets the most relevant frame from the traceback."""
        tb: TracebackType | None = (
            self.cause.__traceback__ if self.cause else self.__traceback__
        )

        while tb is not None:
            frame: FrameType = tb.tb_frame
            lineno: int = tb.tb_lineno
            filename: str = frame.f_code.co_filename

            if "site-packages" not in filename and "lib/python" not in filename:
                code_line: str = linecache.getline(filename, lineno)
                return (filename, lineno, code_line)

            tb = tb.tb_next

        return None

    def _get_code_context(
        self, filename: str, lineno: int, context_lines: int = 3
    ) -> str:
        start: int = max(1, lineno - context_lines)
        end: int = lineno + context_lines

        context: list[str] = []
        for i in range(start, end + 1):
            line: str = linecache.getline(filename, i)
            if line:
                prefix: Literal["  ", ">>"] = ">>" if i == lineno else "  "
                context.append(f"{prefix} {i:4d}: {line.rstrip()}")

        return "\n".join(context)


# ===== CRITICAL ERRORS (HALT EXECUTION) =====
class CriticalExtractionError(ExtractionError):
    def __init__(
        self, message: str, cause: Optional[Exception] = None, **kwargs
    ) -> None:
        super().__init__(f"CRITICAL: {message}", cause=cause, **kwargs)


# ===== RECOVERABLE ERRORS (CONTINUE EXECUTION) =====
class RecoverableExtractionError(ExtractionError):
    def __init__(
        self, message: str, cause: Optional[Exception] = None, **kwargs
    ) -> None:
        super().__init__(f"RECOVERABLE: {message}", cause=cause, **kwargs)
