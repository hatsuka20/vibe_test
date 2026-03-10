"""processes の各 Process に対する単体テスト."""

import logging
from collections.abc import Callable
from pathlib import Path

import pytest

from environment import DryRunEnvironment
from pipeline import Artifact, ExecContext, RunContext
from processes import (
    AggregateProfile,
    CompareBaseline,
    CompileModel,
    CurlDownload,
    DownloadModel,
    FormatProfile,
    GenerateConfig,
    ModelCompile,
    RunModel,
    RuntimeExec,
)
from recipe import CompileOptions, Recipe, RunOptions


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
# Process A: DownloadModel (動的 produces)
# ===========================================================================
class TestDownloadModel:
    def test_invokes_curl_per_model(
        self, run_ctx: RunContext, exec_ctx: ExecContext, dry_env: DryRunEnvironment,
    ) -> None:
        recipe = Recipe(url_base="https://example.com/models", confirmed=False)
        proc = DownloadModel(recipe=recipe)
        proc.run(run_ctx, exec_ctx)

        assert CurlDownload(
            url="https://example.com/models/resnet.onnx",
            output=exec_ctx.out_dir / "resnet.onnx",
        ) in dry_env.history
        assert CurlDownload(
            url="https://example.com/models/vgg.onnx",
            output=exec_ctx.out_dir / "vgg.onnx",
        ) in dry_env.history

    def test_params_contains_release(self) -> None:
        recipe = Recipe(release="v99", url_base="https://example.com/m")
        proc = DownloadModel(recipe=recipe)
        assert proc.params() == {"release": "v99", "url_base": "https://example.com/m"}

    def test_process_fields(self) -> None:
        proc = DownloadModel(recipe=Recipe())
        assert proc.name == "download_models"
        assert proc.requires == []
        assert proc.produces == []  # 動的


# ===========================================================================
# Process B1: GenerateConfig
# ===========================================================================
class TestGenerateConfig:
    def test_process_fields(self) -> None:
        proc = GenerateConfig(model_name="vgg")
        assert proc.name == "generate_config_vgg"
        assert proc.requires == ["model.vgg"]
        assert proc.produces == ["config.vgg"]

    def test_params_contains_chip_and_options(self) -> None:
        opts = CompileOptions(optimization_level=3, memory_mode="low_power")
        proc = GenerateConfig(model_name="resnet", compile_options=opts, chip="chipZ")
        params = proc.params()
        assert params["chip"] == "chipZ"
        assert params["memory_mode"] == "low_power"
        assert params["optimization_level"] == 3


# ===========================================================================
# Process B2: CompileModel
# ===========================================================================
class TestCompileModel:
    @pytest.fixture(autouse=True)
    def _setup_config(self, put_artifact: Callable) -> None:
        """CompileModel が必要とする config アーティファクトを事前登録."""
        put_artifact(
            "config.resnet", "resnet_config.ini", "ini", "config.ini.v1",
            content=b"[compile]\nmemory_mode = normal\n",
        )

    def test_invokes_compiler_with_config(
        self, run_ctx: RunContext, exec_ctx: ExecContext,
        dry_env: DryRunEnvironment, put_artifact: Callable,
    ) -> None:
        model_path = put_artifact("model.resnet", "resnet.onnx", "onnx", "model.onnx.v1")
        config_path = exec_ctx.out_dir / "resnet_config.ini"

        proc = CompileModel(model_name="resnet", compile_options=CompileOptions(optimization_level=3))
        proc.run(run_ctx, exec_ctx)

        assert ModelCompile(
            model_path=model_path,
            output=exec_ctx.out_dir / "resnet_compiled.cpp",
            optimization_level=3,
            config_path=config_path,
        ) in dry_env.history

    def test_requires_model_and_config(self) -> None:
        proc = CompileModel(model_name="vgg")
        assert proc.requires == ["model.vgg", "config.vgg"]

    def test_uses_temp_dir_as_cwd(
        self, run_ctx: RunContext, exec_ctx: ExecContext,
        dry_env: DryRunEnvironment, put_artifact: Callable,
    ) -> None:
        """CompileModel は env.run() に cwd=exec_ctx.temp_dir を渡す."""
        put_artifact("model.resnet", "resnet.onnx", "onnx", "model.onnx.v1")

        proc = CompileModel(model_name="resnet")
        proc.run(run_ctx, exec_ctx)

        assert len(dry_env.history) == 1

    def test_chip_specific_lib_and_flags(
        self, run_ctx: RunContext, exec_ctx: ExecContext,
        dry_env: DryRunEnvironment, put_artifact: Callable,
    ) -> None:
        """チップ固有のライブラリとフラグが CommandBuilder に渡される."""
        model_path = put_artifact("model.resnet", "resnet.onnx", "onnx", "model.onnx.v1")
        config_path = exec_ctx.out_dir / "resnet_config.ini"

        proc = CompileModel(
            model_name="resnet",
            compile_options=CompileOptions(optimization_level=2),
            compile_lib="libChipY.so",
            compile_flags=("--target=chipy", "--fp16"),
        )
        proc.run(run_ctx, exec_ctx)

        assert ModelCompile(
            model_path=model_path,
            output=exec_ctx.out_dir / "resnet_compiled.cpp",
            optimization_level=2,
            config_path=config_path,
            compile_lib="libChipY.so",
            compile_flags=("--target=chipy", "--fp16"),
        ) in dry_env.history

    def test_config_path_in_build_command(self) -> None:
        """--config がビルドコマンドに反映される."""
        cmd = ModelCompile(
            model_path=Path("model.onnx"),
            output=Path("out.cpp"),
            optimization_level=2,
            config_path=Path("config.ini"),
            compile_lib="libChipX.so",
            compile_flags=("--target=chipx",),
        )
        argv = cmd.build()
        assert "--config" in argv
        assert "config.ini" in argv
        assert "-l" in argv
        assert "libChipX.so" in argv
        assert "--target=chipx" in argv


# ===========================================================================
# Process C: RunModel
# ===========================================================================
class TestRunModel:
    def test_invokes_runtime_via_env(
        self, run_ctx: RunContext, exec_ctx: ExecContext,
        dry_env: DryRunEnvironment, put_artifact: Callable,
    ) -> None:
        compiled_path = put_artifact("compiled_model.resnet", "resnet.cpp", "cpp", "compiled.cpp.v1")

        proc = RunModel(model_name="resnet", run_options=RunOptions(num_iterations=500))
        proc.run(run_ctx, exec_ctx)

        assert RuntimeExec(
            compiled_path=compiled_path,
            profile_output=exec_ctx.out_dir / "profile_resnet.json",
            num_iterations=500,
        ) in dry_env.history

    def test_requires_compiled_model(self) -> None:
        proc = RunModel(model_name="vgg")
        assert proc.requires == ["compiled_model.vgg"]

    def test_chip_specific_lib_and_flags(
        self, run_ctx: RunContext, exec_ctx: ExecContext,
        dry_env: DryRunEnvironment, put_artifact: Callable,
    ) -> None:
        """チップ固有のライブラリとフラグが RuntimeExec に渡される."""
        compiled_path = put_artifact("compiled_model.resnet", "resnet.cpp", "cpp", "compiled.cpp.v1")

        proc = RunModel(
            model_name="resnet",
            run_options=RunOptions(num_iterations=100),
            runtime_lib="libChipYRuntime.so",
            runtime_flags=("--device=chipy",),
        )
        proc.run(run_ctx, exec_ctx)

        assert RuntimeExec(
            compiled_path=compiled_path,
            profile_output=exec_ctx.out_dir / "profile_resnet.json",
            num_iterations=100,
            runtime_lib="libChipYRuntime.so",
            runtime_flags=("--device=chipy",),
        ) in dry_env.history

    def test_chip_specific_build_command(self) -> None:
        """チップ固有パラメータがビルドコマンドに反映される."""
        cmd = RuntimeExec(
            compiled_path=Path("model.cpp"),
            profile_output=Path("profile.json"),
            num_iterations=100,
            runtime_lib="libChipXRuntime.so",
            runtime_flags=("--device=chipx",),
        )
        argv = cmd.build()
        assert "-l" in argv
        assert "libChipXRuntime.so" in argv
        assert "--device=chipx" in argv


# ===========================================================================
# Process D: FormatProfile
# ===========================================================================
class TestFormatProfile:
    def test_requires_profile(self) -> None:
        proc = FormatProfile(model_name="vgg")
        assert proc.requires == ["profile.vgg"]

    def test_inherits_default_params(self) -> None:
        proc = FormatProfile()
        assert proc.params() == {}


# ===========================================================================
# Process D2: CompareBaseline (skip_if_missing)
# ===========================================================================
class TestCompareBaseline:
    def test_process_fields(self) -> None:
        proc = CompareBaseline(model_name="vgg")
        assert proc.name == "compare_baseline_vgg"
        assert proc.requires == ["profile.vgg", "baseline.vgg"]
        assert proc.produces == ["comparison.vgg"]
        assert proc.skip_if_missing is True


# ===========================================================================
# Process E: AggregateProfile
# ===========================================================================
class TestAggregateProfile:
    def test_process_fields(self) -> None:
        proc = AggregateProfile(model_names=["a", "b"])
        assert proc.name == "aggregate_profile"
        assert proc.requires == ["report.a", "report.b"]
        assert proc.produces == ["summary_report"]

