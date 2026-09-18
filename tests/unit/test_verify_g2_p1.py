from scripts.verify_g2_p1 import StepResult, build_step_specs, execute_steps


def test_verifier_stops_at_first_failure() -> None:
    specs = build_step_specs(python="python")

    results = execute_steps(
        specs,
        lambda spec: (
            StepResult(spec.name, exit_code=1)
            if spec.name == "pytest-unit"
            else StepResult(spec.name)
        ),
    )

    assert [result.name for result in results] == ["lock-check", "pytest-unit"]


def test_verifier_declares_gpu_phase2_step() -> None:
    specs = build_step_specs(python="python")

    assert any(spec.name == "pytest-gpu-phase2" for spec in specs)
