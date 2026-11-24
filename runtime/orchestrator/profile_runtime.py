"""Explicit runtime profile selection; the historical implementation is unchanged."""


def create_analyzer(profile="original", **kwargs):
    if profile == "original":
        from orchestrator.runtime import RuntimeAnalyzer

        return RuntimeAnalyzer(**kwargs)
    if profile == "final":
        from orchestrator.final_runtime import FinalRuntimeAnalyzer

        return FinalRuntimeAnalyzer(**kwargs)
    raise ValueError("Unknown runtime profile: " + str(profile))
