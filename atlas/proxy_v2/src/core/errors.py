"""Error types for Atlas Proxy v2."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class ErrorType(Enum):
    """Standard error types for API compatibility."""
    # OpenAI error types
    INVALID_REQUEST_ERROR = "invalid_request_error"
    AUTHENTICATION_ERROR = "authentication_error"
    PERMISSION_ERROR = "permission_error"
    NOT_FOUND_ERROR = "not_found_error"
    REQUEST_TIMEOUT = "request_timeout"
    CONFLICT_ERROR = "conflict_error"
    REQUEST_TOO_LARGE = "request_too_large"
    RATE_LIMIT_ERROR = "rate_limit_error"
    INTERNAL_ERROR = "internal_error"
    BAD_GATEWAY = "bad_gateway"
    SERVICE_UNAVAILABLE = "service_unavailable"
    GATEWAY_TIMEOUT = "gateway_timeout"

    # Anthropic error types
    API_ERROR = "api_error"
    OVERLOADED_ERROR = "overloaded_error"

    # Custom error types
    NO_PROVIDER_ERROR = "no_provider_error"
    PROVIDER_ERROR = "provider_error"
    PARSER_ERROR = "parser_error"
    TIMEOUT_ERROR = "timeout_error"


# HTTP status to error type mapping
HTTP_STATUS_MAP = {
    400: ErrorType.INVALID_REQUEST_ERROR,
    401: ErrorType.AUTHENTICATION_ERROR,
    403: ErrorType.PERMISSION_ERROR,
    404: ErrorType.NOT_FOUND_ERROR,
    408: ErrorType.REQUEST_TIMEOUT,
    409: ErrorType.CONFLICT_ERROR,
    413: ErrorType.REQUEST_TOO_LARGE,
    422: ErrorType.INVALID_REQUEST_ERROR,
    429: ErrorType.RATE_LIMIT_ERROR,
    500: ErrorType.INTERNAL_ERROR,
    502: ErrorType.BAD_GATEWAY,
    503: ErrorType.SERVICE_UNAVAILABLE,
    504: ErrorType.GATEWAY_TIMEOUT,
}


def get_error_type(status_code: int) -> ErrorType:
    """Get error type from HTTP status code."""
    return HTTP_STATUS_MAP.get(status_code, ErrorType.API_ERROR)


@dataclass
class APIError:
    """Standard API error."""
    type: ErrorType
    message: str
    status_code: int = 500
    param: Optional[str] = None
    code: Optional[str] = None
    request_id: Optional[str] = None

    def to_openai_dict(self) -> dict[str, Any]:
        """Convert to OpenAI error format."""
        error: dict[str, Any] = {
            "message": self.message,
            "type": self.type.value,
        }
        if self.param:
            error["param"] = self.param
        if self.code:
            error["code"] = self.code
        return {"error": error}

    def to_anthropic_dict(self) -> dict[str, Any]:
        """Convert to Anthropic error format."""
        # Map to Anthropic error types
        anthropic_types = {
            ErrorType.INVALID_REQUEST_ERROR: "invalid_request_error",
            ErrorType.AUTHENTICATION_ERROR: "authentication_error",
            ErrorType.PERMISSION_ERROR: "permission_error",
            ErrorType.NOT_FOUND_ERROR: "not_found_error",
            ErrorType.RATE_LIMIT_ERROR: "rate_limit_error",
            ErrorType.SERVICE_UNAVAILABLE: "overloaded_error",
            ErrorType.OVERLOADED_ERROR: "overloaded_error",
        }

        return {
            "type": "error",
            "error": {
                "type": anthropic_types.get(self.type, "api_error"),
                "message": self.message,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type.value,
            "message": self.message,
            "status_code": self.status_code,
            "param": self.param,
            "code": self.code,
            "request_id": self.request_id,
        }


class ProxyError(Exception):
    """Base exception for proxy errors."""

    def __init__(
        self,
        message: str,
        error_type: ErrorType = ErrorType.API_ERROR,
        status_code: int = 500,
        **kwargs,
    ):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.status_code = status_code
        self.kwargs = kwargs

    def to_api_error(self) -> APIError:
        """Convert to APIError."""
        return APIError(
            type=self.error_type,
            message=self.message,
            status_code=self.status_code,
            **self.kwargs,
        )


class ProviderError(ProxyError):
    """Error from upstream provider."""

    def __init__(self, message: str, provider: str, status_code: int = 502, **kwargs):
        super().__init__(
            message,
            error_type=ErrorType.PROVIDER_ERROR,
            status_code=status_code,
            provider=provider,
            **kwargs,
        )
        self.provider = provider


class NoProviderError(ProxyError):
    """No provider available error."""

    def __init__(self, message: str = "No provider available"):
        super().__init__(
            message,
            error_type=ErrorType.NO_PROVIDER_ERROR,
            status_code=503,
        )


class TimeoutError(ProxyError):
    """Request timeout error."""

    def __init__(self, message: str = "Request timed out"):
        super().__init__(
            message,
            error_type=ErrorType.TIMEOUT_ERROR,
            status_code=504,
        )


class InvalidRequestError(ProxyError):
    """Invalid request error."""

    def __init__(self, message: str, param: Optional[str] = None):
        super().__init__(
            message,
            error_type=ErrorType.INVALID_REQUEST_ERROR,
            status_code=400,
            param=param,
        )


class AuthenticationError(ProxyError):
    """Authentication error."""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(
            message,
            error_type=ErrorType.AUTHENTICATION_ERROR,
            status_code=401,
        )


class RateLimitError(ProxyError):
    """Rate limit error."""

    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(
            message,
            error_type=ErrorType.RATE_LIMIT_ERROR,
            status_code=429,
        )


def create_error_from_response(
    status_code: int,
    body: dict[str, Any],
    provider: Optional[str] = None,
) -> ProxyError:
    """Create appropriate error from upstream response."""

    # Try to extract message from various formats
    message = ""

    # OpenAI format
    if "error" in body:
        err = body["error"]
        if isinstance(err, dict):
            message = err.get("message", "")
            if not message:
                message = err.get("error", {}).get("message", "")

    # Anthropic format
    if "type" in body and body["type"] == "error":
        message = body.get("error", {}).get("message", "")

    # Generic formats
    if not message:
        message = body.get("message", body.get("detail", "Unknown error"))

    error_type = get_error_type(status_code)

    if provider:
        return ProviderError(
            message=message,
            provider=provider,
            status_code=status_code,
        )

    return ProxyError(
        message=message,
        error_type=error_type,
        status_code=status_code,
    )
