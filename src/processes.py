"""パイプラインを構成する Process / CommandBuilder の定義."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from environment import CommandBuilder
from pipeline import (
    ExecContext,
    ProcessBase,
    ProducedArtifact,
    RunContext,
)
from recipe import CompileOptions, Recipe, RunOptions


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
    config_path: Path | None = None
    compiler_path: str = "model-compiler"
    compile_lib: str = ""
    compile_flags: tuple[str, ...] = ()

    def build(self) -> list[str]:
        cmd = [self.compiler_path]
        if self.compile_lib:
            cmd += ["-l", self.compile_lib]
        cmd += list(self.compile_flags)
        if self.config_path:
            cmd += ["--config", str(self.config_path)]
        cmd += [
            f"-O{self.optimization_level}",
            "-o", str(self.output),
            str(self.model_path),
        ]
        return cmd


@dataclass(frozen=True)
class RuntimeExec(CommandBuilder):
    compiled_path: Path
    profile_output: Path
    num_iterations: int
    runtime_path: str = "model-runtime"
    runtime_lib: str = ""
    runtime_flags: tuple[str, ...] = ()

    def build(self) -> list[str]:
        cmd = [self.runtime_path]
        if self.runtime_lib:
            cmd += ["-l", self.runtime_lib]
        cmd += list(self.runtime_flags)
        cmd += [
            "--profile", str(self.profile_output),
            "--iterations", str(self.num_iterations),
            str(self.compiled_path),
        ]
        return cmd


# ---------------------------------------------------------------------------
# Process A: リモートサーバからモデルをダウンロードする (mock)
#   動的 produces: 実行時にモデル名を発見し model.<name> を生成
# ---------------------------------------------------------------------------
@dataclass
class DownloadModel(ProcessBase):
    recipe: Recipe = field(default_factory=Recipe)
    recipe_path: Path | None = None
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        self.name = "download_models"
        self.produces = []  # 動的 produces

    def params(self) -> dict:
        return {"release": self.recipe.release, "url_base": self.recipe.url_base}

    def _discover_models(self) -> list[str]:
        """モデル名を発見する (mock)."""
        return ["resnet", "vgg"]

    def run(self, ctx: RunContext, exec_ctx: ExecContext) -> dict[str, ProducedArtifact]:
        # confirmed=True → レシピのモデルリストを正として使用
        # confirmed=False → 実行時に発見して反映
        if self.recipe.models_confirmed():
            model_names = self.recipe.model_names()
            exec_ctx.logger.info("[A] レシピのモデルリストを使用: %s", model_names)
        else:
            model_names = self._discover_models()

        url_base = self.recipe.url_base
        result = {}
        for name in model_names:
            model_path = exec_ctx.out_path(f"{name}.onnx")
            cmd = CurlDownload(url=f"{url_base}/{name}.onnx", output=model_path)
            exec_ctx.logger.info("[A] モデルをダウンロード中: %s", cmd.url)
            exec_ctx.env.run(cmd)

            # mock: 実コマンドの代わりにダミーファイルを生成
            if exec_ctx.env.executes:
                model_path.write_bytes(b"\x00MOCK_" + name.encode() + b"_WEIGHTS" * 64)

            exec_ctx.logger.info("[A] ダウンロード完了 -> %s", model_path)
            result[f"model.{name}"] = ProducedArtifact(model_path, "onnx", "model.onnx.v1")

        # confirmed でない場合のみレシピに書き戻す
        if self.recipe_path and not self.recipe.models_confirmed():
            if self.recipe.populate_models(model_names):
                self.recipe.save(self.recipe_path)
                exec_ctx.logger.info(
                    "[A] レシピにモデル名を反映: %s", self.recipe_path,
                )

        return result


# ---------------------------------------------------------------------------
# Process B1: チップ固有のコンパイル設定ファイルを生成する (mock)
#   CompileOptions の内容をチップごとのフォーマットで config ファイルに変換
# ---------------------------------------------------------------------------
_CONFIG_GENERATORS: dict[str, str] = {
    "chipX": "ini",
    "chipY": "json",
    "chipZ": "json",
}


@dataclass
class GenerateConfig(ProcessBase):
    model_name: str = "default"
    compile_options: CompileOptions = field(default_factory=CompileOptions)
    chip: str = "chipX"
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        self.name = f"generate_config_{self.model_name}"
        self.requires = [f"model.{self.model_name}"]
        self.produces = [f"config.{self.model_name}"]

    def params(self) -> dict:
        return {
            **self.compile_options.model_dump(),
            "chip": self.chip,
        }

    def _generate_ini(self) -> str:
        """chipX 向け: INI 形式の config を生成."""
        lines = [
            "[compile]",
            f"memory_mode = {self.compile_options.memory_mode}",
        ]
        if self.compile_options.quantization_bits is not None:
            lines.append(f"quantization_bits = {self.compile_options.quantization_bits}")
        return "\n".join(lines) + "\n"

    def _generate_json(self) -> str:
        """chipY/chipZ 向け: JSON 形式の config を生成."""
        data: dict = {
            "memory_mode": self.compile_options.memory_mode,
        }
        if self.compile_options.quantization_bits is not None:
            data["quantization"] = {"bits": self.compile_options.quantization_bits}
        return json.dumps(data, indent=2)

    def run(self, ctx: RunContext, exec_ctx: ExecContext) -> dict[str, ProducedArtifact]:
        fmt = _CONFIG_GENERATORS.get(self.chip, "json")

        if fmt == "ini":
            content = self._generate_ini()
            ext = "ini"
        else:
            content = self._generate_json()
            ext = "json"

        config_path = exec_ctx.out_path(f"{self.model_name}_config.{ext}")
        if exec_ctx.env.executes:
            config_path.write_text(content, encoding="utf-8")

        exec_ctx.logger.info("[B1] config 生成完了 -> %s (%s形式)", config_path, fmt)
        return {
            f"config.{self.model_name}": ProducedArtifact(config_path, ext, f"config.{fmt}.v1"),
        }


# ---------------------------------------------------------------------------
# Process B2: モデルを中間形式 (cpp) にコンパイルする (mock)
# ---------------------------------------------------------------------------
@dataclass
class CompileModel(ProcessBase):
    model_name: str = "default"
    compile_options: CompileOptions = field(default_factory=CompileOptions)
    compiler_path: str = "model-compiler"
    compile_lib: str = ""
    compile_flags: tuple[str, ...] = ()
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        self.name = f"compile_{self.model_name}"
        self.requires = [f"model.{self.model_name}", f"config.{self.model_name}"]
        self.produces = [f"compiled_model.{self.model_name}"]

    def params(self) -> dict:
        return {
            "optimization_level": self.compile_options.optimization_level,
            "compiler_path": self.compiler_path,
            "compile_lib": self.compile_lib,
            "compile_flags": list(self.compile_flags),
        }

    def run(self, ctx: RunContext, exec_ctx: ExecContext) -> dict[str, ProducedArtifact]:
        model_art = ctx.get(f"model.{self.model_name}")
        config_art = ctx.get(f"config.{self.model_name}")
        cpp_path = exec_ctx.out_path(f"{self.model_name}_compiled.cpp")

        cmd = ModelCompile(
            model_path=model_art.path,
            output=cpp_path,
            optimization_level=self.compile_options.optimization_level,
            config_path=config_art.path,
            compiler_path=self.compiler_path,
            compile_lib=self.compile_lib,
            compile_flags=self.compile_flags,
        )
        exec_ctx.logger.info(
            "[B2] モデルをコンパイル中: %s (O%d, config=%s)",
            model_art.path, self.compile_options.optimization_level, config_art.path,
        )
        exec_ctx.env.run(cmd, cwd=exec_ctx.temp_dir)

        # mock: 実コマンドの代わりにダミーファイルを生成
        if exec_ctx.env.executes:
            cpp_content = f"""\
// Auto-generated from {model_art.path}
// config = {config_art.path}
// optimization_level = {self.compile_options.optimization_level}
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

        exec_ctx.logger.info("[B2] コンパイル完了 -> %s", cpp_path)
        return {
            f"compiled_model.{self.model_name}": ProducedArtifact(cpp_path, "cpp", "compiled.cpp.v1"),
        }


# ---------------------------------------------------------------------------
# Process C: Runtime で中間形式を実行し、プロファイルを取得する (mock)
# ---------------------------------------------------------------------------
@dataclass
class RunModel(ProcessBase):
    model_name: str = "default"
    run_options: RunOptions = field(default_factory=RunOptions)
    runtime_path: str = "model-runtime"
    runtime_lib: str = ""
    runtime_flags: tuple[str, ...] = ()
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        self.name = f"run_{self.model_name}"
        self.requires = [f"compiled_model.{self.model_name}"]
        self.produces = [f"profile.{self.model_name}"]

    def params(self) -> dict:
        return {
            **self.run_options.model_dump(),
            "runtime_path": self.runtime_path,
            "runtime_lib": self.runtime_lib,
            "runtime_flags": list(self.runtime_flags),
        }

    def run(self, ctx: RunContext, exec_ctx: ExecContext) -> dict[str, ProducedArtifact]:
        compiled_art = ctx.get(f"compiled_model.{self.model_name}")
        profile_path = exec_ctx.out_path(f"profile_{self.model_name}.json")

        cmd = RuntimeExec(
            compiled_path=compiled_art.path,
            profile_output=profile_path,
            num_iterations=self.run_options.num_iterations,
            runtime_path=self.runtime_path,
            runtime_lib=self.runtime_lib,
            runtime_flags=self.runtime_flags,
        )
        exec_ctx.logger.info(
            "[C] Runtime で実行中: %s (%d iterations)", compiled_art.path, self.run_options.num_iterations,
        )
        exec_ctx.env.run(cmd)

        # mock: 実コマンドの代わりにダミーファイルを生成
        if exec_ctx.env.executes:
            profile_data = {
                "source": str(compiled_art.path),
                "iterations": self.run_options.num_iterations,
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
        if not exec_ctx.env.executes:
            exec_ctx.logger.info("[D] DryRun時はskip: %s", self.model_name)
            return {}

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

        report_path = exec_ctx.out_path(f"report_{self.model_name}.txt")
        report_path.write_text("\n".join(lines), encoding="utf-8")

        exec_ctx.logger.info("[D] レポート生成完了 -> %s", report_path)
        return {
            f"report.{self.model_name}": ProducedArtifact(report_path, "txt", "report.human.v1"),
        }


# ---------------------------------------------------------------------------
# Process D2: ベースラインとの比較レポートを生成する (skip_if_missing の例)
#   baseline.<name> がなければプロセスごとスキップされ、あれば比較結果を出力する
# ---------------------------------------------------------------------------
@dataclass
class CompareBaseline(ProcessBase):
    model_name: str = "default"
    version: str = "1.0.0"
    skip_if_missing: bool = True

    def __post_init__(self) -> None:
        self.name = f"compare_baseline_{self.model_name}"
        self.requires = [f"profile.{self.model_name}", f"baseline.{self.model_name}"]
        self.produces = [f"comparison.{self.model_name}"]

    def run(self, ctx: RunContext, exec_ctx: ExecContext) -> dict[str, ProducedArtifact]:
        if not exec_ctx.env.executes:
            exec_ctx.logger.info("[D2] DryRun時はskip: %s", self.model_name)
            return {}

        profile_art = ctx.get(f"profile.{self.model_name}")
        baseline_art = ctx.get(f"baseline.{self.model_name}")

        profile = json.loads(profile_art.path.read_text(encoding="utf-8"))
        baseline = json.loads(baseline_art.path.read_text(encoding="utf-8"))

        cur_lat = profile["latency_ms"]["mean"]
        base_lat = baseline["latency_ms"]["mean"]
        diff_pct = (cur_lat - base_lat) / base_lat * 100

        lines = [
            f"Baseline Comparison: {self.model_name}",
            f"  baseline latency : {base_lat:.2f} ms",
            f"  current  latency : {cur_lat:.2f} ms",
            f"  diff             : {diff_pct:+.1f}%",
        ]

        comparison_path = exec_ctx.out_path(f"comparison_{self.model_name}.txt")
        comparison_path.write_text("\n".join(lines), encoding="utf-8")

        exec_ctx.logger.info("[D2] ベースライン比較完了 -> %s (%+.1f%%)", comparison_path, diff_pct)
        return {
            f"comparison.{self.model_name}": ProducedArtifact(comparison_path, "txt", "comparison.v1"),
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
        if not exec_ctx.env.executes:
            exec_ctx.logger.info("[E] DryRun時はskip: %s", self.model_names)
            return {}

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

        summary_path = exec_ctx.out_path("summary_report.txt")
        summary_path.write_text("\n".join(lines), encoding="utf-8")

        exec_ctx.logger.info("[E] 集約完了 -> %s", summary_path)
        return {"summary_report": ProducedArtifact(summary_path, "txt", "report.summary.v1")}


# ---------------------------------------------------------------------------
# Process F: コンパイル済みモデルを様々な反復回数でベンチマークする (fan-out 例)
#   kwargs_factory が list[dict] を返すことで variant ごとに複数プロセスに展開される
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkModel(ProcessBase):
    model_name: str = "default"
    num_iterations: int = 100
    runtime_path: str = "model-runtime"
    runtime_lib: str = ""
    runtime_flags: tuple[str, ...] = ()
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        self.name = f"benchmark_{self.model_name}_n{self.num_iterations}"
        self.requires = [f"compiled_model.{self.model_name}"]
        self.produces = [f"benchmark.{self.model_name}.n{self.num_iterations}"]

    def params(self) -> dict:
        return {
            "num_iterations": self.num_iterations,
            "runtime_path": self.runtime_path,
            "runtime_lib": self.runtime_lib,
            "runtime_flags": list(self.runtime_flags),
        }

    def run(self, ctx: RunContext, exec_ctx: ExecContext) -> dict[str, ProducedArtifact]:
        compiled_art = ctx.get(f"compiled_model.{self.model_name}")

        cmd = RuntimeExec(
            compiled_path=compiled_art.path,
            profile_output=exec_ctx.out_path("_discard"),  # dummy
            num_iterations=self.num_iterations,
            runtime_path=self.runtime_path,
            runtime_lib=self.runtime_lib,
            runtime_flags=self.runtime_flags,
        )
        exec_ctx.logger.info(
            "[F] ベンチマーク実行中: %s (%d iterations)",
            compiled_art.path, self.num_iterations,
        )
        exec_ctx.env.run(cmd)

        # mock
        bench_path = exec_ctx.out_path(
            f"benchmark_{self.model_name}_n{self.num_iterations}.json",
        )
        if exec_ctx.env.executes:
            import random
            bench_data = {
                "model": self.model_name,
                "iterations": self.num_iterations,
                "latency_ms": round(2.0 + random.random() * 3, 2),
                "throughput": round(400 + random.random() * 200, 1),
            }
            bench_path.write_text(json.dumps(bench_data, indent=2), encoding="utf-8")

        exec_ctx.logger.info("[F] ベンチマーク完了 -> %s", bench_path)
        key = f"benchmark.{self.model_name}.n{self.num_iterations}"
        return {key: ProducedArtifact(bench_path, "json", "benchmark.v1")}
