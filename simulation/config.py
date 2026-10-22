# File to hold the constants

from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class SimulationConfig:
    """Scenario assumptions for the emergency-department simulation."""

    num_doctors: int = 4
    num_nurses: int = 13
    num_beds: int = 70
    acuity_probabilities: dict[str, float] = field(
        default_factory=lambda: {
            "emergent": 0.24,
            "urgent": 0.47,
            "non-urgent": 0.29,
        }
    )
    triage_min_minutes: float = 3.0
    triage_max_minutes: float = 7.0
    treatment_min_minutes: float = 30.0
    treatment_mode_minutes: float = 42.0
    treatment_max_minutes: float = 55.0
    diversion_occupancy_threshold: float | None = 0.85
    wait_target_minutes: float = 45.0

    def __post_init__(self) -> None:
        for name in ("num_doctors", "num_nurses", "num_beds"):
            value = getattr(self, name)
            if not math.isfinite(value) or int(value) != value or value < 1:
                raise ValueError(f"{name} must be at least 1.")

        if not self.acuity_probabilities:
            raise ValueError("acuity_probabilities cannot be empty.")
        probabilities = list(self.acuity_probabilities.values())
        if any(not math.isfinite(p) or p < 0 for p in probabilities):
            raise ValueError("Acuity probabilities must be finite and nonnegative.")
        if abs(sum(self.acuity_probabilities.values()) - 1.0) > 1e-9:
            raise ValueError("Acuity probabilities must sum to 1.")

        durations = (
            self.triage_min_minutes,
            self.triage_max_minutes,
            self.treatment_min_minutes,
            self.treatment_mode_minutes,
            self.treatment_max_minutes,
            self.wait_target_minutes,
        )
        if any(not math.isfinite(value) for value in durations):
            raise ValueError("Service durations and wait target must be finite.")
        if self.triage_min_minutes < 0 or self.triage_max_minutes < self.triage_min_minutes:
            raise ValueError("Triage duration bounds must be nonnegative and ordered.")
        if not (
            0 <= self.treatment_min_minutes
            <= self.treatment_mode_minutes
            <= self.treatment_max_minutes
        ):
            raise ValueError("Treatment duration bounds must be nonnegative and ordered.")
        if self.diversion_occupancy_threshold is not None and not (
            math.isfinite(self.diversion_occupancy_threshold)
            and 0 <= self.diversion_occupancy_threshold <= 1
        ):
            raise ValueError("Diversion occupancy threshold must be between 0 and 1.")
        if self.wait_target_minutes < 0:
            raise ValueError("wait_target_minutes cannot be negative.")
