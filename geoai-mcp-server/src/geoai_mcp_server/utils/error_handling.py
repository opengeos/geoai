"""Error handling utilities for GeoAI MCP Server.

Provides structured error types and formatting for MCP responses.
"""

from typing import Optional, Any
from enum import Enum


class ErrorCategory(str, Enum):
    """Error categories for structured error responses."""
    CLIENT_ERROR = "client_error"  # Invalid inputs, missing files, bad parameters
    SERVER_ERROR = "server_error"  # Internal failures, OOM, model load failures
    EXTERNAL_ERROR = "external_error"  # Network failures, API rate limits
    TIMEOUT_ERROR = "timeout_error"  # Operation exceeded time limit


class GeoAIError(Exception):
    """Base exception for GeoAI MCP Server errors."""

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.SERVER_ERROR,
        details: Optional[dict[str, Any]] = None,
        suggestion: Optional[str] = None,
        retry_allowed: bool = False,
    ):
        """Initialize GeoAI error.

        Args:
            message: Human-readable error message
            category: Error category for classification
            details: Additional details for debugging
            suggestion: Suggested fix or next steps
            retry_allowed: Whether retrying might succeed
        """
        super().__init__(message)
        self.message = message
        self.category = category
        self.details = details or {}
        self.suggestion = suggestion
        self.retry_allowed = retry_allowed


class InputValidationError(GeoAIError):
    """Error raised for invalid input parameters."""

    def __init__(
        self,
        message: str,
        parameter_name: Optional[str] = None,
        received_value: Optional[Any] = None,
        expected: Optional[str] = None,
    ):
        details = {}
        if parameter_name:
            details["parameter"] = parameter_name
        if received_value is not None:
            details["received"] = str(received_value)[:100]  # Truncate long values
        if expected:
            details["expected"] = expected

        suggestion = None
        if expected:
            suggestion = f"Expected {expected}"
        if parameter_name:
            suggestion = f"Check the '{parameter_name}' parameter. {suggestion or ''}"

        super().__init__(
            message=message,
            category=ErrorCategory.CLIENT_ERROR,
            details=details,
            suggestion=suggestion,
            retry_allowed=False,
        )


class FileAccessError(GeoAIError):
    """Error raised for file system access issues."""

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        operation: str = "access",
    ):
        details = {}
        if file_path:
            details["file_path"] = file_path
        details["operation"] = operation

        suggestion = None
        if operation == "read":
            suggestion = "Ensure the file exists in the input directory and is readable."
        elif operation == "write":
            suggestion = "Ensure the output directory is writable."
        elif operation == "access":
            suggestion = "Check that the file path is within the allowed workspace directories."

        super().__init__(
            message=message,
            category=ErrorCategory.CLIENT_ERROR,
            details=details,
            suggestion=suggestion,
            retry_allowed=False,
        )


class ModelLoadError(GeoAIError):
    """Error raised when model loading fails."""

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        reason: Optional[str] = None,
    ):
        details = {}
        if model_name:
            details["model"] = model_name
        if reason:
            details["reason"] = reason

        super().__init__(
            message=message,
            category=ErrorCategory.SERVER_ERROR,
            details=details,
            suggestion="Try using a different model or check GPU memory availability.",
            retry_allowed=True,
        )


class ProcessingError(GeoAIError):
    """Error raised during image/data processing."""

    def __init__(
        self,
        message: str,
        stage: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        error_details = details or {}
        if stage:
            error_details["stage"] = stage

        super().__init__(
            message=message,
            category=ErrorCategory.SERVER_ERROR,
            details=error_details,
            suggestion="Try with a smaller image or different parameters.",
            retry_allowed=True,
        )


class TimeoutError(GeoAIError):
    """Error raised when operation exceeds time limit."""

    def __init__(
        self,
        message: str,
        timeout_seconds: int,
        operation: Optional[str] = None,
    ):
        details = {
            "timeout_seconds": timeout_seconds,
        }
        if operation:
            details["operation"] = operation

        super().__init__(
            message=message,
            category=ErrorCategory.TIMEOUT_ERROR,
            details=details,
            suggestion="Try with a smaller image, reduce tile size, or increase timeout.",
            retry_allowed=True,
        )


class ExternalServiceError(GeoAIError):
    """Error raised for external service failures (network, APIs)."""

    def __init__(
        self,
        message: str,
        service: Optional[str] = None,
        status_code: Optional[int] = None,
    ):
        details = {}
        if service:
            details["service"] = service
        if status_code:
            details["status_code"] = status_code

        super().__init__(
            message=message,
            category=ErrorCategory.EXTERNAL_ERROR,
            details=details,
            suggestion="Check network connectivity and try again later.",
            retry_allowed=True,
        )


def format_error_response(error: Exception) -> dict[str, Any]:
    """Format an exception into a structured error response.

    Args:
        error: The exception to format

    Returns:
        Dictionary with error details suitable for MCP response
    """
    if isinstance(error, GeoAIError):
        response = {
            "success": False,
            "error": {
                "message": error.message,
                "category": error.category.value,
                "details": error.details,
            }
        }
        if error.suggestion:
            response["error"]["suggestion"] = error.suggestion
        if error.retry_allowed:
            response["error"]["retry_allowed"] = True
        return response

    # Generic error handling
    return {
        "success": False,
        "error": {
            "message": str(error),
            "category": ErrorCategory.SERVER_ERROR.value,
            "details": {"type": type(error).__name__},
            "suggestion": "An unexpected error occurred. Please check the logs for details.",
        }
    }
