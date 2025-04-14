from benchmarking.environment_suites.cause_effect_pairs import (
    BenchmarkingCauseEffectPairs,
)
from benchmarking.environment_suites.gymnasium import BenchmarkingGymnasiumEnvs
from benchmarking.environment_suites.vmas import BenchmarkingVmasEnvs

ENVIRONMENT_SUITES = {
    "gymnasium": BenchmarkingGymnasiumEnvs,
    "vmas": BenchmarkingVmasEnvs,
    "cause_effect_pairs": BenchmarkingCauseEffectPairs,
}
