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
    # Replaced triangular bounds with statistical parameters for a long-tail distribution
    treatment_mean_minutes: float = 42.0
    treatment_std_dev_minutes: float = 15.0
    diversion_occupancy_threshold: float | None = 0.85
    wait_target_minutes: float = 45.0

    def __post_init__(self) -> None:
        # Validate integer resources
        for name in ("num_doctors", "num_nurses", "num_beds"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer of at least 1.")

        # Validate probabilities
        if not self.acuity_probabilities:
            raise ValueError("acuity_probabilities cannot be empty.")
        
        probabilities = list(self.acuity_probabilities.values())
        if any(not math.isfinite(p) or p < 0 for p in probabilities):
            raise ValueError("Acuity probabilities must be finite and nonnegative.")
        
        if abs(sum(probabilities) - 1.0) > 1e-9:
            raise ValueError("Acuity probabilities must sum to exactly 1.0.")

        # Validate time durations (updated for new treatment parameters)
        durations = (
            self.triage_min_minutes,
            self.triage_max_minutes,
            self.treatment_mean_minutes,
            self.treatment_std_dev_minutes,
            self.wait_target_minutes,
        )
        if any(not math.isfinite(value) for value in durations):
            raise ValueError("Service durations and wait target must be finite.")
            
        if self.triage_min_minutes < 0 or self.triage_max_minutes < self.triage_min_minutes:
            raise ValueError("Triage duration bounds must be nonnegative and ordered correctly.")
            
        # New validation for treatment statistical parameters
        if self.treatment_mean_minutes < 0 or self.treatment_std_dev_minutes < 0:
            raise ValueError("Treatment mean and standard deviation must be nonnegative.")
            
        if self.diversion_occupancy_threshold is not None:
            if not math.isfinite(self.diversion_occupancy_threshold) or not (0 <= self.diversion_occupancy_threshold <= 1):
                raise ValueError("Diversion occupancy threshold must be a finite number between 0 and 1.")
                
        if self.wait_target_minutes < 0:
            raise ValueError("wait_target_minutes cannot be negative.")

# Create a baseline configuration object for the Streamlit app to import
DEFAULT_CONFIG = SimulationConfig()