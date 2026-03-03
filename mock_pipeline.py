"""仮想的な4段パイプラインのモック実装."""

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from environment import CommandBuilder, DryRunEnvironment
from pipeline import (
    ExecContext,
    Pipeline,
    ProcessBase,
    ProducedArtifact,
    RunContext,
)


# ---------------------------------------------------------------------------
# CommandBuilder 群
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CurlDownload(CommandBuilder):
    url: str
    output: Path

    def build(self) -> list[str]:
        return ["curl", "-sSL", "-o", str(self.output), self.url]


@dataclass(frozen=True)
class ModelCompile(CommandBuilder):
    model_path: Path
    output: Path
    optimization_level: int

    def build(self) -> list[str]:
        return [
            "model-compiler",
            f"-O{self.optimization_level}",
            "-o", str(self.output),
            str(self.model_path),
        ]


@dataclass(frozen=True)
class RuntimeExec(CommandBuilder):
    compiled_path: Path
    profile_output: Path
    num_iterations: int

    def build(self) -> list[str]:
        return [
            "model-runtime",
            "--profile", str(self.profile_output),
            "--iterations", str(self.num_iterations),
            str(self.compiled_path),
        ]


# ---------------------------------------------------------------------------
# Process A: リモートサーバからモデルをダウンロードする (mock)
# ---------------------------------------------------------------------------
@dataclass
class DownloadModel(ProcessBase):
    name: str = "download_model"
    produces: list[str] = field(default_factory=lambda: ["model"])
    version: str = "1.0.0"

    url: str = "https://example.com/models/resnet50.onnx"

    def params(self) -> dict:
        return {"url": self.url}

    def run(self, ctx: RunContext, exec_ctx: ExecContext) -> dict[str, ProducedArtifact]:
        model_path = exec_ctx.out_dir / "resnet50.onnx"

        cmd = CurlDownload(url=self.url, output=model_path)
        exec_ctx.logger.info("[A] モデルをダウンロード中: %s", self.url)
        exec_ctx.env.run(cmd)

        # mock: 実コマンドの代わりにダミーファイルを生成
        model_path.write_bytes(b"\x00MOCK_ONNX_MODEL_WEIGHTS" * 64)

        exec_ctx.logger.info("[A] ダウンロード完了 -> %s", model_path)
        return {"model": ProducedArtifact(model_path, "onnx", "model.onnx.v1")}


# ---------------------------------------------------------------------------
# Process B: モデルを中間形式 (cpp) にコンパイルする (mock)
# ---------------------------------------------------------------------------
@dataclass
class CompileModel(ProcessBase):
    name: str = "compile_model"
    requires: list[str] = field(default_factory=lambda: ["model"])
    produces: list[str] = field(default_factory=lambda: ["compiled_model"])
    version: str = "1.0.0"

    optimization_level: int = 2

    def params(self) -> dict:
        return {"optimization_level": self.optimization_level}

    def run(self, ctx: RunContext, exec_ctx: ExecContext) -> dict[str, ProducedArtifact]:
        model_art = ctx.get("model")
        cpp_path = exec_ctx.out_dir / "model_compiled.cpp"

        cmd = ModelCompile(
            model_path=Path(model_art.path),
            output=cpp_path,
            optimization_level=self.optimization_level,
        )
        exec_ctx.logger.info("[B] モデルをコンパイル中: %s (O%d)", model_art.path, self.optimization_level)
        exec_ctx.env.run(cmd)

        # mock: 実コマンドの代わりにダミーファイルを生成
        cpp_content = f"""\
// Auto-generated from {model_art.path}
// optimization_level = {self.optimization_level}
#include <cstdint>

namespace model {{
  static const float weights[] = {{0.1f, 0.2f, 0.3f}};

  void infer(const float* input, float* output, int n) {{
    for (int i = 0; i < n; ++i) {{
      output[i] = input[i] * weights[i % 3];
    }}
  }}
}}
"""
        cpp_path.write_text(cpp_content, encoding="utf-8")

        exec_ctx.logger.info("[B] コンパイル完了 -> %s", cpp_path)
        return {"compiled_model": ProducedArtifact(cpp_path, "cpp", "compiled.cpp.v1")}


# ---------------------------------------------------------------------------
# Process C: Runtime で中間形式を実行し、プロファイルを取得する (mock)
# ---------------------------------------------------------------------------
@dataclass
class RunModel(ProcessBase):
    name: str = "run_model"
    requires: list[str] = field(default_factory=lambda: ["compiled_model"])
    produces: list[str] = field(default_factory=lambda: ["profile"])
    version: str = "1.0.0"

    num_iterations: int = 100

    def params(self) -> dict:
        return {"num_iterations": self.num_iterations}

    def run(self, ctx: RunContext, exec_ctx: ExecContext) -> dict[str, ProducedArtifact]:
        compiled_art = ctx.get("compiled_model")
        profile_path = exec_ctx.out_dir / "profile.json"

        cmd = RuntimeExec(
            compiled_path=Path(compiled_art.path),
            profile_output=profile_path,
            num_iterations=self.num_iterations,
        )
        exec_ctx.logger.info("[C] Runtime で実行中: %s (%d iterations)", compiled_art.path, self.num_iterations)
        exec_ctx.env.run(cmd)

        # mock: 実コマンドの代わりにダミーファイルを生成
        profile_data = {
            "source": compiled_art.path,
            "iterations": self.num_iterations,
            "latency_ms": {"min": 1.2, "max": 5.8, "mean": 2.4, "p99": 4.9},
            "throughput_items_per_sec": 416.7,
            "memory_peak_mb": 128.5,
            "ops": [
                {"name": "conv2d_1", "time_ms": 0.8, "memory_mb": 32.0},
                {"name": "relu_1", "time_ms": 0.1, "memory_mb": 0.5},
                {"name": "pool_1", "time_ms": 0.3, "memory_mb": 8.0},
                {"name": "fc_1", "time_ms": 0.5, "memory_mb": 16.0},
            ],
        }
        profile_path.write_text(json.dumps(profile_data, indent=2), encoding="utf-8")

        exec_ctx.logger.info("[C] 実行完了 -> %s", profile_path)
        return {"profile": ProducedArtifact(profile_path, "json", "profile.runtime.v1")}


# ---------------------------------------------------------------------------
# Process D: プロファイルを人間が読みやすいレポートに整形する (mock)
#   subprocess 不要のため CommandBuilder / Environment は使用しない
# ---------------------------------------------------------------------------
@dataclass
class FormatProfile(ProcessBase):
    name: str = "format_profile"
    requires: list[str] = field(default_factory=lambda: ["profile"])
    produces: list[str] = field(default_factory=lambda: ["report"])
    version: str = "1.0.0"

    def run(self, ctx: RunContext, exec_ctx: ExecContext) -> dict[str, ProducedArtifact]:
        profile_art = ctx.get("profile")
        exec_ctx.logger.info("[D] プロファイルを整形中: %s", profile_art.path)

        profile = json.loads(Path(profile_art.path).read_text(encoding="utf-8"))
        lat = profile["latency_ms"]

        lines = [
            "=" * 60,
            "  Model Performance Report",
            "=" * 60,
            "",
            f"  Source      : {profile['source']}",
            f"  Iterations  : {profile['iterations']}",
            "",
            "  Latency (ms)",
            f"    min  : {lat['min']:.2f}",
            f"    max  : {lat['max']:.2f}",
            f"    mean : {lat['mean']:.2f}",
            f"    p99  : {lat['p99']:.2f}",
            "",
            f"  Throughput  : {profile['throughput_items_per_sec']:.1f} items/sec",
            f"  Peak Memory : {profile['memory_peak_mb']:.1f} MB",
            "",
            "  Per-Op Breakdown",
            "  " + "-" * 46,
            f"  {'Op':<16} {'Time (ms)':>10} {'Memory (MB)':>12}",
            "  " + "-" * 46,
        ]
        for op in profile["ops"]:
            lines.append(f"  {op['name']:<16} {op['time_ms']:>10.2f} {op['memory_mb']:>12.1f}")
        lines += ["  " + "-" * 46, "=" * 60, ""]

        report_path = exec_ctx.out_dir / "report.txt"
        report_path.write_text("\n".join(lines), encoding="utf-8")

        exec_ctx.logger.info("[D] レポート生成完了 -> %s", report_path)
        return {"report": ProducedArtifact(report_path, "txt", "report.human.v1")}


# ---------------------------------------------------------------------------
# エントリポイント
# ---------------------------------------------------------------------------
def main() -> None:
    class Args(argparse.Namespace):
        experiment_name: str

    parser = argparse.ArgumentParser(description="モックパイプラインの実行")
    parser.add_argument("experiment_name", help="実験名 (experiments/<name> に出力)")
    args = parser.parse_args(namespace=Args())

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("mock_pipeline")

    base_dir = Path("experiments") / args.experiment_name
    run_dir = base_dir / "run"
    out_dir = base_dir / "out"
    temp_dir = base_dir / "tmp"
    for d in (out_dir, temp_dir):
        d.mkdir(parents=True, exist_ok=True)

    env = DryRunEnvironment()
    ctx = RunContext.load(run_dir=run_dir)
    exec_ctx = ExecContext(out_dir=out_dir, temp_dir=temp_dir, logger=logger, env=env)

    pipeline = Pipeline([
        DownloadModel(),
        CompileModel(),
        RunModel(),
        FormatProfile(),
    ])

    ctx = pipeline.run(ctx, exec_ctx)

    logger.info("Manifest: %s", ctx.manifest_path)
    for cmd in env.history:
        logger.info("Command: %s", " ".join(cmd.build()))


if __name__ == "__main__":
    main()
