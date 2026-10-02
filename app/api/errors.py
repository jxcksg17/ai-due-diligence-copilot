"""Production-facing service errors with no transport coupling."""


class ProductionServiceError(RuntimeError):
    """Base class for operational failures exposed through the API layer."""


class ModelUnavailableError(ProductionServiceError):
    pass


class ModelTimeoutError(ProductionServiceError):
    pass


class AIConcurrencyLimitError(ProductionServiceError):
    pass


class UnsupportedOperationError(ProductionServiceError):
    pass
