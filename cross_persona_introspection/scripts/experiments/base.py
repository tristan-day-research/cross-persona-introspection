"""Base class for experiments."""

from abc import ABC, abstractmethod
from cross_persona_introspection.schemas import RunConfig


class BaseExperiment(ABC):
    """Minimal base for PSM introspection experiments."""

    def __init__(self, config: RunConfig):
        self.config = config

    @abstractmethod
    def setup(self) -> None:
        """Load model, tasks, personas."""
        ...

    @abstractmethod
    def run(self) -> None:
        """Execute all trials."""
        ...

    @abstractmethod
    def evaluate(self) -> dict:
        """Compute aggregate metrics. Returns summary dict."""
        ...

    @abstractmethod
    def save_results(self) -> str:
        """Save results and return output path."""
        ...
