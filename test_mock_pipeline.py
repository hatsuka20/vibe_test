"""mock_pipeline の各 Process に対する単体テスト."""

import json
import logging
from pathlib import Path

import pytest

from environment import DryRunEnvironment
from main import Artifact, ExecContext, RunContext
from mock_pipeline import (
    CompileModel,
    CurlDownload,
    DownloadModel,
    FormatProfile,
    ModelCompile,
    RunModel,
    RuntimeExec,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def run_ctx(tmp_path: Path) -> RunContext:
    return RunContext(run_dir=tmp_path / "run")


@pytest.fixture()
def dry_env() -> DryRunEnvironment:
    return DryRunEnvironment()


@pytest.fixture()
def exec_ctx(tmp_path: Path, dry_env: DryRunEnvironment) -> ExecContext:
    out_dir = tmp_path / "out"
    temp_dir = tmp_path / "tmp"
    out_dir.mkdir()
    temp_dir.mkdir()
    return ExecContext(
        out_dir=out_dir,
        temp_dir=temp_dir,
        logger=logging.getLogger("test"),
        env=dry_env,
    )


def _put_dummy_artifact(ctx: RunContext, key: str, path: Path, fmt: str, schema: str) -> None:
    """テスト用のダミー Artifact を RunContext に登録する."""
    ctx.artifacts[key] = Artifact(
        key=key,
        path=str(path),
        format=fmt,
        schema=schema,
        producer="test",
        cache_key="dummy",
        sha256="dummy",
    )


# ===========================================================================
# CommandBuilder 単体テスト
# ===========================================================================
class TestCurlDownload:
    def test_build(self, tmp_path: Path) -> None:
        cmd = CurlDownload(url="https://example.com/m.onnx", output=tmp_path / "m.onnx")
        args = cmd.build()
        assert args[0] == "curl"
        assert "https://example.com/m.onnx" in args
        assert str(tmp_path / "m.onnx") in args


class TestModelCompile:
    def test_build(self, tmp_path: Path) -> None:
        cmd = ModelCompile(
            model_path=tmp_path / "in.onnx",
            output=tmp_path / "out.cpp",
            optimization_level=3,
        )
        args = cmd.build()
        assert args[0] == "model-compiler"
        assert "-O3" in args
        assert str(tmp_path / "in.onnx") in args
        assert str(tmp_path / "out.cpp") in args


class TestRuntimeExec:
    def test_build(self, tmp_path: Path) -> None:
        cmd = RuntimeExec(
            compiled_path=tmp_path / "model.cpp",
            profile_output=tmp_path / "profile.json",
            num_iterations=200,
        )
        args = cmd.build()
        assert args[0] == "model-runtime"
        assert "200" in args
        assert str(tmp_path / "model.cpp") in args


# ===========================================================================
# Process A: DownloadModel
# ===========================================================================
class TestDownloadModel:
    def test_produces_model_file(self, run_ctx: RunContext, exec_ctx: ExecContext) -> None:
        proc = DownloadModel()
        result = proc.run(run_ctx, exec_ctx)

        assert "model" in result
        path, fmt, schema = result["model"]
        assert path.exists()
        assert path.stat().st_size > 0
        assert fmt == "onnx"
        assert schema == "model.onnx.v1"

    def test_invokes_curl_via_env(self, run_ctx: RunContext, exec_ctx: ExecContext, dry_env: DryRunEnvironment) -> None:
        proc = DownloadModel(url="https://example.com/test.onnx")
        proc.run(run_ctx, exec_ctx)

        assert len(dry_env.history) == 1
        cmd = dry_env.history[0]
        assert cmd[0] == "curl"
        assert "https://example.com/test.onnx" in cmd

    def test_params_contains_url(self) -> None:
        proc = DownloadModel(url="https://example.com/custom.onnx")
        assert proc.params() == {"url": "https://example.com/custom.onnx"}

    def test_protocol_fields(self) -> None:
        proc = DownloadModel()
        assert proc.name == "download_model"
        assert proc.requires == []
        assert proc.produces == ["model"]


# ===========================================================================
# Process B: CompileModel
# ===========================================================================
class TestCompileModel:
    def test_produces_cpp_file(self, run_ctx: RunContext, exec_ctx: ExecContext) -> None:
        model_path = exec_ctx.out_dir / "dummy.onnx"
        model_path.write_bytes(b"MOCK")
        _put_dummy_artifact(run_ctx, "model", model_path, "onnx", "model.onnx.v1")

        proc = CompileModel()
        result = proc.run(run_ctx, exec_ctx)

        assert "compiled_model" in result
        path, fmt, schema = result["compiled_model"]
        assert path.exists()
        assert fmt == "cpp"
        assert schema == "compiled.cpp.v1"

    def test_invokes_compiler_via_env(
        self, run_ctx: RunContext, exec_ctx: ExecContext, dry_env: DryRunEnvironment,
    ) -> None:
        model_path = exec_ctx.out_dir / "dummy.onnx"
        model_path.write_bytes(b"MOCK")
        _put_dummy_artifact(run_ctx, "model", model_path, "onnx", "model.onnx.v1")

        proc = CompileModel(optimization_level=3)
        proc.run(run_ctx, exec_ctx)

        assert len(dry_env.history) == 1
        cmd = dry_env.history[0]
        assert cmd[0] == "model-compiler"
        assert "-O3" in cmd

    def test_cpp_contains_source_reference(self, run_ctx: RunContext, exec_ctx: ExecContext) -> None:
        model_path = exec_ctx.out_dir / "dummy.onnx"
        model_path.write_bytes(b"MOCK")
        _put_dummy_artifact(run_ctx, "model", model_path, "onnx", "model.onnx.v1")

        proc = CompileModel(optimization_level=3)
        result = proc.run(run_ctx, exec_ctx)

        content = result["compiled_model"][0].read_text(encoding="utf-8")
        assert str(model_path) in content
        assert "optimization_level = 3" in content

    def test_requires_model(self) -> None:
        proc = CompileModel()
        assert proc.requires == ["model"]


# ===========================================================================
# Process C: RunModel
# ===========================================================================
class TestRunModel:
    def test_produces_valid_profile_json(self, run_ctx: RunContext, exec_ctx: ExecContext) -> None:
        cpp_path = exec_ctx.out_dir / "model.cpp"
        cpp_path.write_text("// mock", encoding="utf-8")
        _put_dummy_artifact(run_ctx, "compiled_model", cpp_path, "cpp", "compiled.cpp.v1")

        proc = RunModel()
        result = proc.run(run_ctx, exec_ctx)

        assert "profile" in result
        path, fmt, schema = result["profile"]
        assert path.exists()
        assert fmt == "json"
        assert schema == "profile.runtime.v1"

        profile = json.loads(path.read_text(encoding="utf-8"))
        assert "latency_ms" in profile
        assert "throughput_items_per_sec" in profile
        assert "memory_peak_mb" in profile
        assert "ops" in profile
        assert len(profile["ops"]) > 0

    def test_invokes_runtime_via_env(
        self, run_ctx: RunContext, exec_ctx: ExecContext, dry_env: DryRunEnvironment,
    ) -> None:
        cpp_path = exec_ctx.out_dir / "model.cpp"
        cpp_path.write_text("// mock", encoding="utf-8")
        _put_dummy_artifact(run_ctx, "compiled_model", cpp_path, "cpp", "compiled.cpp.v1")

        proc = RunModel(num_iterations=500)
        proc.run(run_ctx, exec_ctx)

        assert len(dry_env.history) == 1
        cmd = dry_env.history[0]
        assert cmd[0] == "model-runtime"
        assert "500" in cmd

    def test_profile_reflects_iterations(self, run_ctx: RunContext, exec_ctx: ExecContext) -> None:
        cpp_path = exec_ctx.out_dir / "model.cpp"
        cpp_path.write_text("// mock", encoding="utf-8")
        _put_dummy_artifact(run_ctx, "compiled_model", cpp_path, "cpp", "compiled.cpp.v1")

        proc = RunModel(num_iterations=500)
        result = proc.run(run_ctx, exec_ctx)

        profile = json.loads(result["profile"][0].read_text(encoding="utf-8"))
        assert profile["iterations"] == 500

    def test_requires_compiled_model(self) -> None:
        proc = RunModel()
        assert proc.requires == ["compiled_model"]


# ===========================================================================
# Process D: FormatProfile
# ===========================================================================
class TestFormatProfile:
    @pytest.fixture()
    def _setup_profile(self, run_ctx: RunContext, exec_ctx: ExecContext) -> None:
        profile_data = {
            "source": "model.cpp",
            "iterations": 100,
            "latency_ms": {"min": 1.0, "max": 5.0, "mean": 2.0, "p99": 4.0},
            "throughput_items_per_sec": 400.0,
            "memory_peak_mb": 100.0,
            "ops": [
                {"name": "conv2d_1", "time_ms": 0.8, "memory_mb": 32.0},
            ],
        }
        profile_path = exec_ctx.out_dir / "profile.json"
        profile_path.write_text(json.dumps(profile_data), encoding="utf-8")
        _put_dummy_artifact(run_ctx, "profile", profile_path, "json", "profile.runtime.v1")

    @pytest.mark.usefixtures("_setup_profile")
    def test_produces_human_readable_report(self, run_ctx: RunContext, exec_ctx: ExecContext) -> None:
        proc = FormatProfile()
        result = proc.run(run_ctx, exec_ctx)

        assert "report" in result
        path, fmt, schema = result["report"]
        assert path.exists()
        assert fmt == "txt"
        assert schema == "report.human.v1"

    @pytest.mark.usefixtures("_setup_profile")
    def test_report_contains_key_sections(self, run_ctx: RunContext, exec_ctx: ExecContext) -> None:
        proc = FormatProfile()
        result = proc.run(run_ctx, exec_ctx)

        report = result["report"][0].read_text(encoding="utf-8")
        assert "Model Performance Report" in report
        assert "Latency" in report
        assert "Throughput" in report
        assert "Peak Memory" in report
        assert "Per-Op Breakdown" in report
        assert "conv2d_1" in report

    @pytest.mark.usefixtures("_setup_profile")
    def test_does_not_invoke_env(
        self, run_ctx: RunContext, exec_ctx: ExecContext, dry_env: DryRunEnvironment,
    ) -> None:
        proc = FormatProfile()
        proc.run(run_ctx, exec_ctx)
        assert len(dry_env.history) == 0

    def test_requires_profile(self) -> None:
        proc = FormatProfile()
        assert proc.requires == ["profile"]
