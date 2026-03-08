"""パイプラインを構成する Process / CommandBuilder の定義."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from environment import CommandBuilder
from pipeline import (
    ExecContext,
    ProcessBase,
    ProducedArtifact,
    RunContext,
)

if TYPE_CHECKING:
    from recipe import Recipe


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
#   動的 produces: 実行時にモデル名を発見し model.<name> を生成
# ---------------------------------------------------------------------------
@dataclass
class DownloadModel(ProcessBase):
    release: str = "v50"
    url_base: str = "https://example.com/models"
    recipe: Recipe | None = None
    recipe_path: Path | None = None
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        self.name = "download_models"
        self.produces = []  # 動的 produces

    def params(self) -> dict:
        return {"release": self.release, "url_base": self.url_base}

    def run(self, ctx: RunContext, exec_ctx: ExecContext) -> dict[str, ProducedArtifact]:
        # Mock: 実行時にモデルを発見
        model_names = ["resnet", "vgg"]

        result = {}
        for name in model_names:
            model_path = exec_ctx.out_dir / f"{name}.onnx"
            cmd = CurlDownload(url=f"{self.url_base}/{name}.onnx", output=model_path)
            exec_ctx.logger.info("[A] モデルをダウンロード中: %s", cmd.url)
            exec_ctx.env.run(cmd)

            # mock: 実コマンドの代わりにダミーファイルを生成
            model_path.write_bytes(b"\x00MOCK_" + name.encode() + b"_WEIGHTS" * 64)

            exec_ctx.logger.info("[A] ダウンロード完了 -> %s", model_path)
            result[f"model.{name}"] = ProducedArtifact(model_path, "onnx", "model.onnx.v1")

        # レシピにモデル名を書き戻す
        if self.recipe and self.recipe_path:
            if self.recipe.populate_models(model_names):
                self.recipe.save(self.recipe_path)
                exec_ctx.logger.info(
                    "[A] レシピにモデル名を反映: %s", self.recipe_path,
                )

        return result


# ---------------------------------------------------------------------------
# Process B: モデルを中間形式 (cpp) にコンパイルする (mock)
# ---------------------------------------------------------------------------
@dataclass
class CompileModel(ProcessBase):
    model_name: str = "default"
    optimization_level: int = 2
    recipe: Recipe | None = None
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        self.name = f"compile_{self.model_name}"
        self.requires = [f"model.{self.model_name}"]
        self.produces = [f"compiled_model.{self.model_name}"]
        if self.recipe:
            opts = self.recipe.resolve_compile_options(self.model_name)
            self.optimization_level = opts.optimization_level

    def params(self) -> dict:
        return {"optimization_level": self.optimization_level}

    def run(self, ctx: RunContext, exec_ctx: ExecContext) -> dict[str, ProducedArtifact]:
        model_art = ctx.get(f"model.{self.model_name}")
        cpp_path = exec_ctx.out_dir / f"{self.model_name}_compiled.cpp"

        cmd = ModelCompile(
            model_path=model_art.path,
            output=cpp_path,
            optimization_level=self.optimization_level,
        )
        exec_ctx.logger.info(
            "[B] モデルをコンパイル中: %s (O%d)", model_art.path, self.optimization_level,
        )
        exec_ctx.env.run(cmd, cwd=exec_ctx.temp_dir)

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
        return {
            f"compiled_model.{self.model_name}": ProducedArtifact(cpp_path, "cpp", "compiled.cpp.v1"),
        }


# ---------------------------------------------------------------------------
# Process C: Runtime で中間形式を実行し、プロファイルを取得する (mock)
# ---------------------------------------------------------------------------
@dataclass
class RunModel(ProcessBase):
    model_name: str = "default"
    num_iterations: int = 100
    recipe: Recipe | None = None
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        self.name = f"run_{self.model_name}"
        self.requires = [f"compiled_model.{self.model_name}"]
        self.produces = [f"profile.{self.model_name}"]
        if self.recipe:
            opts = self.recipe.resolve_run_options(self.model_name)
            self.num_iterations = opts.num_iterations

    def params(self) -> dict:
        return {"num_iterations": self.num_iterations}

    def run(self, ctx: RunContext, exec_ctx: ExecContext) -> dict[str, ProducedArtifact]:
        compiled_art = ctx.get(f"compiled_model.{self.model_name}")
        profile_path = exec_ctx.out_dir / f"profile_{self.model_name}.json"

        cmd = RuntimeExec(
            compiled_path=compiled_art.path,
            profile_output=profile_path,
            num_iterations=self.num_iterations,
        )
        exec_ctx.logger.info(
            "[C] Runtime で実行中: %s (%d iterations)", compiled_art.path, self.num_iterations,
        )
        exec_ctx.env.run(cmd)

        # mock: 実コマンドの代わりにダミーファイルを生成
        profile_data = {
            "source": str(compiled_art.path),
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
        return {
            f"profile.{self.model_name}": ProducedArtifact(profile_path, "json", "profile.runtime.v1"),
        }


# ---------------------------------------------------------------------------
# Process D: プロファイルを人間が読みやすいレポートに整形する (mock)
#   subprocess 不要のため CommandBuilder / Environment は使用しない
# ---------------------------------------------------------------------------
@dataclass
class FormatProfile(ProcessBase):
    model_name: str = "default"
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        self.name = f"format_{self.model_name}"
        self.requires = [f"profile.{self.model_name}"]
        self.produces = [f"report.{self.model_name}"]

    def run(self, ctx: RunContext, exec_ctx: ExecContext) -> dict[str, ProducedArtifact]:
        profile_art = ctx.get(f"profile.{self.model_name}")
        exec_ctx.logger.info("[D] プロファイルを整形中: %s", profile_art.path)

        profile = json.loads(profile_art.path.read_text(encoding="utf-8"))
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

        report_path = exec_ctx.out_dir / f"report_{self.model_name}.txt"
        report_path.write_text("\n".join(lines), encoding="utf-8")

        exec_ctx.logger.info("[D] レポート生成完了 -> %s", report_path)
        return {
            f"report.{self.model_name}": ProducedArtifact(report_path, "txt", "report.human.v1"),
        }


# ---------------------------------------------------------------------------
# Process E: 全モデルのレポートを集約してサマリー比較テーブルを生成する
# ---------------------------------------------------------------------------
@dataclass
class AggregateProfile(ProcessBase):
    model_names: list[str] = field(default_factory=list)
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        self.name = "aggregate_profile"
        self.requires = [f"report.{m}" for m in self.model_names]
        self.produces = ["summary_report"]

    def run(self, ctx: RunContext, exec_ctx: ExecContext) -> dict[str, ProducedArtifact]:
        exec_ctx.logger.info("[E] レポートを集約中: %s", self.model_names)

        # 各モデルの profile を読み取り (report ではなく profile から数値取得)
        summaries = []
        for name in self.model_names:
            report_art = ctx.get(f"report.{name}")
            report_text = report_art.path.read_text(encoding="utf-8")
            summaries.append({"model": name, "report": report_text})

        # サマリーテーブル生成
        lines = [
            "=" * 60,
            "  Aggregate Summary",
            "=" * 60,
            "",
            f"  Models: {', '.join(self.model_names)}",
            "",
        ]
        for s in summaries:
            lines.append(f"  --- {s['model']} ---")
            lines.append(s["report"])
            lines.append("")
        lines.append("=" * 60)

        summary_path = exec_ctx.out_dir / "summary_report.txt"
        summary_path.write_text("\n".join(lines), encoding="utf-8")

        exec_ctx.logger.info("[E] 集約完了 -> %s", summary_path)
        return {"summary_report": ProducedArtifact(summary_path, "txt", "report.summary.v1")}
