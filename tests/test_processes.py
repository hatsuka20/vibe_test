"""processes の各 Process に対する単体テスト."""

import json
import logging
from collections.abc import Callable
from pathlib import Path

import pytest

from environment import DryRunEnvironment
from pipeline import Artifact, ExecContext, RunContext
from processes import (
    CompileModel,
    CurlDownload,
    DownloadModel,
    FormatProfile,
    ModelCompile,
    RunModel,
    RuntimeExec,
)


# NOTE: CommandBuilder.build() の単体テストは省略。
# Process テストが DryRunEnvironment 経由で正しい CommandBuilder の構築を検証している。


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


@pytest.fixture()
def put_artifact(
    run_ctx: RunContext, exec_ctx: ExecContext,
) -> Callable[..., Path]:
    """テスト用アーティファクトをファイル作成 + RunContext 登録する fixture factory."""

    def _put(
        key: str, filename: str, fmt: str, schema: str, *, content: bytes = b"MOCK",
    ) -> Path:
        path = exec_ctx.out_dir / filename
        path.write_bytes(content)
        run_ctx.artifacts[key] = Artifact(
            key=key,
            path=path,
            format=fmt,
            schema=schema,
            producer="test",
            cache_key="dummy",
            sha256="dummy",
        )
        return path

    return _put


# ===========================================================================
# Process A: DownloadModel
# ===========================================================================
class TestDownloadModel:
    def test_produces_model_file(self, run_ctx: RunContext, exec_ctx: ExecContext) -> None:
        proc = DownloadModel()
        result = proc.run(run_ctx, exec_ctx)

        assert "model" in result
        art = result["model"]
        assert art.path.exists()
        assert art.path.stat().st_size > 0
        assert art.format == "onnx"

    def test_invokes_curl_via_env(self, run_ctx: RunContext, exec_ctx: ExecContext, dry_env: DryRunEnvironment) -> None:
        proc = DownloadModel(url="https://example.com/test.onnx")
        proc.run(run_ctx, exec_ctx)

        assert CurlDownload(
            url="https://example.com/test.onnx",
            output=exec_ctx.out_dir / "resnet50.onnx",
        ) in dry_env.history

    def test_params_contains_url(self) -> None:
        proc = DownloadModel(url="https://example.com/custom.onnx")
        assert proc.params() == {"url": "https://example.com/custom.onnx"}

    def test_process_fields(self) -> None:
        proc = DownloadModel()
        assert proc.name == "download_model"
        assert proc.requires == []
        assert proc.produces == ["model"]


# ===========================================================================
# Process B: CompileModel
# ===========================================================================
class TestCompileModel:
    def test_produces_cpp_file(
        self, run_ctx: RunContext, exec_ctx: ExecContext, put_artifact: Callable,
    ) -> None:
        put_artifact("model", "dummy.onnx", "onnx", "model.onnx.v1")

        proc = CompileModel()
        result = proc.run(run_ctx, exec_ctx)

        assert "compiled_model" in result
        art = result["compiled_model"]
        assert art.path.exists()
        assert art.format == "cpp"

    def test_invokes_compiler_via_env(
        self, run_ctx: RunContext, exec_ctx: ExecContext,
        dry_env: DryRunEnvironment, put_artifact: Callable,
    ) -> None:
        model_path = put_artifact("model", "dummy.onnx", "onnx", "model.onnx.v1")

        proc = CompileModel(optimization_level=3)
        proc.run(run_ctx, exec_ctx)

        assert ModelCompile(
            model_path=model_path,
            output=exec_ctx.out_dir / "model_compiled.cpp",
            optimization_level=3,
        ) in dry_env.history

    def test_cpp_contains_source_reference(
        self, run_ctx: RunContext, exec_ctx: ExecContext, put_artifact: Callable,
    ) -> None:
        model_path = put_artifact("model", "dummy.onnx", "onnx", "model.onnx.v1")

        proc = CompileModel(optimization_level=3)
        result = proc.run(run_ctx, exec_ctx)

        content = result["compiled_model"].path.read_text(encoding="utf-8")
        assert str(model_path) in content
        assert "optimization_level = 3" in content

    def test_requires_model(self) -> None:
        proc = CompileModel()
        assert proc.requires == ["model"]


# ===========================================================================
# Process C: RunModel
# ===========================================================================
class TestRunModel:
    def test_produces_valid_profile_json(
        self, run_ctx: RunContext, exec_ctx: ExecContext, put_artifact: Callable,
    ) -> None:
        put_artifact("compiled_model", "model.cpp", "cpp", "compiled.cpp.v1")

        proc = RunModel()
        result = proc.run(run_ctx, exec_ctx)

        assert "profile" in result
        art = result["profile"]
        assert art.path.exists()
        assert art.format == "json"

        profile = json.loads(art.path.read_text(encoding="utf-8"))
        assert "latency_ms" in profile
        assert "throughput_items_per_sec" in profile
        assert "memory_peak_mb" in profile
        assert "ops" in profile
        assert len(profile["ops"]) > 0

    def test_invokes_runtime_via_env(
        self, run_ctx: RunContext, exec_ctx: ExecContext,
        dry_env: DryRunEnvironment, put_artifact: Callable,
    ) -> None:
        compiled_path = put_artifact("compiled_model", "model.cpp", "cpp", "compiled.cpp.v1")

        proc = RunModel(num_iterations=500)
        proc.run(run_ctx, exec_ctx)

        assert RuntimeExec(
            compiled_path=compiled_path,
            profile_output=exec_ctx.out_dir / "profile.json",
            num_iterations=500,
        ) in dry_env.history

    def test_profile_reflects_iterations(
        self, run_ctx: RunContext, exec_ctx: ExecContext, put_artifact: Callable,
    ) -> None:
        put_artifact("compiled_model", "model.cpp", "cpp", "compiled.cpp.v1")

        proc = RunModel(num_iterations=500)
        result = proc.run(run_ctx, exec_ctx)

        profile = json.loads(result["profile"].path.read_text(encoding="utf-8"))
        assert profile["iterations"] == 500

    def test_requires_compiled_model(self) -> None:
        proc = RunModel()
        assert proc.requires == ["compiled_model"]


# ===========================================================================
# Process D: FormatProfile
# ===========================================================================
class TestFormatProfile:
    @pytest.fixture()
    def _setup_profile(self, put_artifact: Callable, exec_ctx: ExecContext) -> None:
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
        put_artifact(
            "profile", "profile.json", "json", "profile.runtime.v1",
            content=json.dumps(profile_data).encode(),
        )

    @pytest.mark.usefixtures("_setup_profile")
    def test_produces_human_readable_report(self, run_ctx: RunContext, exec_ctx: ExecContext) -> None:
        proc = FormatProfile()
        result = proc.run(run_ctx, exec_ctx)

        assert "report" in result
        art = result["report"]
        assert art.path.exists()
        assert art.format == "txt"

    @pytest.mark.usefixtures("_setup_profile")
    def test_report_contains_key_sections(self, run_ctx: RunContext, exec_ctx: ExecContext) -> None:
        proc = FormatProfile()
        result = proc.run(run_ctx, exec_ctx)

        report = result["report"].path.read_text(encoding="utf-8")
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

    def test_inherits_default_params(self) -> None:
        proc = FormatProfile()
        assert proc.params() == {}
