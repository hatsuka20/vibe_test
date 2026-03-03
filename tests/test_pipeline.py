"""pipeline モジュール (Pipeline, compute_cache_key, RunContext) の単体テスト."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from environment import DryRunEnvironment
from typing import Any

from pipeline import (
    Artifact,
    ExecContext,
    OptionalInput,
    Pipeline,
    ProcessBase,
    ProducedArtifact,
    RunContext,
    compute_cache_key,
)


# ---------------------------------------------------------------------------
# テスト用の軽量 Process
# ---------------------------------------------------------------------------
@dataclass
class StubProcess(ProcessBase):
    """テスト用の最小 Process. run() はファイルを1つ生成するだけ."""

    name: str = "stub"
    produces: list[str] = field(default_factory=lambda: ["out"])
    version: str = "1.0.0"

    content: bytes = b"STUB"
    call_count: int = field(default=0, init=False, repr=False)

    def params(self) -> dict[str, Any]:
        return {"content": self.content.hex()}

    def run(self, ctx: RunContext, exec_ctx: ExecContext) -> dict[str, ProducedArtifact]:
        self.call_count += 1
        path = exec_ctx.out_dir / f"{self.name}.dat"
        path.write_bytes(self.content)
        return {"out": ProducedArtifact(path, "bin", "stub.v1")}


@dataclass
class TwoOutputProcess(ProcessBase):
    """produces が2つある Process."""

    name: str = "two_out"
    produces: list[str] = field(default_factory=lambda: ["a", "b"])
    version: str = "1.0.0"

    def run(self, ctx: RunContext, exec_ctx: ExecContext) -> dict[str, ProducedArtifact]:
        pa = exec_ctx.out_dir / "a.dat"
        pb = exec_ctx.out_dir / "b.dat"
        pa.write_bytes(b"A")
        pb.write_bytes(b"B")
        return {
            "a": ProducedArtifact(pa, "bin", "a.v1"),
            "b": ProducedArtifact(pb, "bin", "b.v1"),
        }


@dataclass
class BadProcess(ProcessBase):
    """produces 宣言と run() の戻り値が食い違う Process."""

    name: str = "bad"
    produces: list[str] = field(default_factory=lambda: ["expected"])
    version: str = "1.0.0"

    def run(self, ctx: RunContext, exec_ctx: ExecContext) -> dict[str, ProducedArtifact]:
        path = exec_ctx.out_dir / "wrong.dat"
        path.write_bytes(b"X")
        return {"wrong_key": ProducedArtifact(path, "bin", "bad.v1")}


@dataclass
class DependentProcess(ProcessBase):
    """requires で前段の成果物に依存する Process."""

    name: str = "dependent"
    requires: list[str] = field(default_factory=lambda: ["out"])
    produces: list[str] = field(default_factory=lambda: ["derived"])
    version: str = "1.0.0"

    def run(self, ctx: RunContext, exec_ctx: ExecContext) -> dict[str, ProducedArtifact]:
        prev = ctx.get("out")
        data = prev.path.read_bytes()
        path = exec_ctx.out_dir / "derived.dat"
        path.write_bytes(data + b"_derived")
        return {"derived": ProducedArtifact(path, "bin", "derived.v1")}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def run_ctx(tmp_path: Path) -> RunContext:
    return RunContext(run_dir=tmp_path / "run")


@pytest.fixture()
def exec_ctx(tmp_path: Path) -> ExecContext:
    out_dir = tmp_path / "out"
    temp_dir = tmp_path / "tmp"
    out_dir.mkdir()
    temp_dir.mkdir()
    return ExecContext(
        out_dir=out_dir,
        temp_dir=temp_dir,
        logger=logging.getLogger("test"),
        env=DryRunEnvironment(),
    )


# ===========================================================================
# Pipeline.__init__ バリデーション
# ===========================================================================
class TestPipelineInit:
    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="empty name"):
            Pipeline([StubProcess(name="")])

    def test_duplicate_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Duplicate"):
            Pipeline([StubProcess(name="a"), StubProcess(name="a")])

    def test_valid_processes(self) -> None:
        pipeline = Pipeline([StubProcess(name="a"), StubProcess(name="b")])
        assert len(pipeline.processes) == 2


# ===========================================================================
# Pipeline.run — 基本動作
# ===========================================================================
class TestPipelineRun:
    def test_single_process(self, run_ctx: RunContext, exec_ctx: ExecContext) -> None:
        pipeline = Pipeline([StubProcess()])
        ctx = pipeline.run(run_ctx, exec_ctx)

        assert "out" in ctx.artifacts
        art = ctx.artifacts["out"]
        assert art.path.exists()
        assert art.producer == "stub"

    def test_two_stage_pipeline(self, run_ctx: RunContext, exec_ctx: ExecContext) -> None:
        pipeline = Pipeline([StubProcess(), DependentProcess()])
        ctx = pipeline.run(run_ctx, exec_ctx)

        assert "out" in ctx.artifacts
        assert "derived" in ctx.artifacts
        assert ctx.artifacts["derived"].path.read_bytes() == b"STUB_derived"

    def test_multi_output_process(self, run_ctx: RunContext, exec_ctx: ExecContext) -> None:
        pipeline = Pipeline([TwoOutputProcess()])
        ctx = pipeline.run(run_ctx, exec_ctx)

        assert "a" in ctx.artifacts
        assert "b" in ctx.artifacts


# ===========================================================================
# Pipeline.run — キャッシュ
# ===========================================================================
class TestPipelineCache:
    def test_second_run_is_cache_hit(self, run_ctx: RunContext, exec_ctx: ExecContext) -> None:
        proc = StubProcess()
        pipeline = Pipeline([proc])
        pipeline.run(run_ctx, exec_ctx)
        assert proc.call_count == 1

        pipeline.run(run_ctx, exec_ctx)
        # キャッシュヒット → run() が再度呼ばれない
        assert proc.call_count == 1

    def test_force_reruns_despite_cache(self, run_ctx: RunContext, exec_ctx: ExecContext) -> None:
        proc = StubProcess()
        pipeline = Pipeline([proc])
        pipeline.run(run_ctx, exec_ctx)
        assert proc.call_count == 1

        pipeline.run(run_ctx, exec_ctx, force_processes=["stub"])
        # force → キャッシュがあっても再実行される
        assert proc.call_count == 2

    def test_param_change_invalidates_cache(self, run_ctx: RunContext, exec_ctx: ExecContext) -> None:
        pipeline1 = Pipeline([StubProcess(content=b"V1")])
        pipeline1.run(run_ctx, exec_ctx)
        sha_v1 = run_ctx.artifacts["out"].sha256

        pipeline2 = Pipeline([StubProcess(content=b"V2")])
        pipeline2.run(run_ctx, exec_ctx)
        sha_v2 = run_ctx.artifacts["out"].sha256

        assert sha_v1 != sha_v2


# ===========================================================================
# Pipeline.run — from_process / force_processes バリデーション
# ===========================================================================
class TestPipelineRunValidation:
    def test_unknown_from_process_raises(self, run_ctx: RunContext, exec_ctx: ExecContext) -> None:
        pipeline = Pipeline([StubProcess()])
        with pytest.raises(ValueError, match="Unknown from_process"):
            pipeline.run(run_ctx, exec_ctx, from_process="nonexistent")

    def test_unknown_force_processes_raises(self, run_ctx: RunContext, exec_ctx: ExecContext) -> None:
        pipeline = Pipeline([StubProcess()])
        with pytest.raises(ValueError, match="Unknown force_processes"):
            pipeline.run(run_ctx, exec_ctx, force_processes=["nonexistent"])

    def test_from_process_skips_earlier(self, run_ctx: RunContext, exec_ctx: ExecContext) -> None:
        # StubProcess の出力を事前登録して dependent から開始
        stub_path = exec_ctx.out_dir / "stub.dat"
        stub_path.write_bytes(b"PRESTAGED")
        from pipeline import sha256_file
        run_ctx.artifacts["out"] = Artifact(
            key="out",
            path=stub_path,
            format="bin",
            schema="stub.v1",
            producer="stub",
            cache_key="pre",
            sha256=sha256_file(stub_path),
        )

        pipeline = Pipeline([StubProcess(), DependentProcess()])
        ctx = pipeline.run(run_ctx, exec_ctx, from_process="dependent")

        assert "derived" in ctx.artifacts
        assert ctx.artifacts["derived"].path.read_bytes() == b"PRESTAGED_derived"

    def test_produces_mismatch_raises(self, run_ctx: RunContext, exec_ctx: ExecContext) -> None:
        pipeline = Pipeline([BadProcess()])
        with pytest.raises(RuntimeError, match="produces mismatch"):
            pipeline.run(run_ctx, exec_ctx)


# ===========================================================================
# compute_cache_key
# ===========================================================================
class TestComputeCacheKey:
    def test_deterministic(self, run_ctx: RunContext) -> None:
        proc = StubProcess()
        key1 = compute_cache_key(proc, run_ctx)
        key2 = compute_cache_key(proc, run_ctx)
        assert key1 == key2

    def test_different_version_different_key(self, run_ctx: RunContext) -> None:
        key1 = compute_cache_key(StubProcess(version="1.0.0"), run_ctx)
        key2 = compute_cache_key(StubProcess(version="2.0.0"), run_ctx)
        assert key1 != key2

    def test_different_name_different_key(self, run_ctx: RunContext) -> None:
        key1 = compute_cache_key(StubProcess(name="a"), run_ctx)
        key2 = compute_cache_key(StubProcess(name="b"), run_ctx)
        assert key1 != key2


# ===========================================================================
# RunContext — save / load ラウンドトリップ
# ===========================================================================
class TestRunContextPersistence:
    def test_roundtrip(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        ctx = RunContext(run_dir=run_dir)

        art_path = tmp_path / "artifact.dat"
        art_path.write_bytes(b"data")

        ctx.put(Artifact(
            key="item",
            path=art_path,
            format="bin",
            schema="test.v1",
            producer="test",
            cache_key="ck",
            sha256="abc",
        ))

        loaded = RunContext.load(run_dir=run_dir)
        assert "item" in loaded.artifacts
        art = loaded.artifacts["item"]
        assert art.key == "item"
        assert art.path == art_path
        assert art.format == "bin"
        assert art.sha256 == "abc"

    def test_load_without_manifest(self, tmp_path: Path) -> None:
        ctx = RunContext.load(run_dir=tmp_path / "empty_run")
        assert ctx.artifacts == {}


# ===========================================================================
# RunContext — get / get_optional
# ===========================================================================
class TestRunContextAccess:
    def test_get_missing_key_raises(self, run_ctx: RunContext) -> None:
        with pytest.raises(KeyError, match="Missing artifact"):
            run_ctx.get("nonexistent")

    def test_get_optional_returns_none(self, run_ctx: RunContext) -> None:
        assert run_ctx.get_optional("nonexistent") is None


# ===========================================================================
# compute_cache_key — optional 入力
# ===========================================================================
@dataclass
class OptionalProcess(ProcessBase):
    """optional 入力を持つ Process."""

    name: str = "opt_proc"
    produces: list[str] = field(default_factory=lambda: ["out"])
    optional: list[OptionalInput] = field(
        default_factory=lambda: [
            OptionalInput(key="hint", affects_cache=True),
            OptionalInput(key="debug_flag", affects_cache=False),
        ]
    )
    version: str = "1.0.0"

    def run(self, ctx: RunContext, exec_ctx: ExecContext) -> dict[str, ProducedArtifact]:
        path = exec_ctx.out_dir / "out.dat"
        path.write_bytes(b"OPT")
        return {"out": ProducedArtifact(path, "bin", "opt.v1")}


class TestComputeCacheKeyOptional:
    def test_optional_absent(self, run_ctx: RunContext) -> None:
        """optional 入力が存在しない場合もキャッシュキーが計算できる."""
        proc = OptionalProcess()
        key = compute_cache_key(proc, run_ctx)
        assert isinstance(key, str) and len(key) == 64

    def test_optional_present_changes_key(self, run_ctx: RunContext, tmp_path: Path) -> None:
        """affects_cache=True の optional 入力が存在するとキーが変わる."""
        proc = OptionalProcess()
        key_without = compute_cache_key(proc, run_ctx)

        hint_path = tmp_path / "hint.dat"
        hint_path.write_bytes(b"HINT")
        run_ctx.artifacts["hint"] = Artifact(
            key="hint", path=hint_path, format="bin", schema="hint.v1",
            producer="test", cache_key="x", sha256="abc",
        )
        key_with = compute_cache_key(proc, run_ctx)

        assert key_without != key_with

    def test_affects_cache_false_ignored(self, run_ctx: RunContext, tmp_path: Path) -> None:
        """affects_cache=False の optional 入力はキーに影響しない."""
        proc = OptionalProcess()
        key_before = compute_cache_key(proc, run_ctx)

        flag_path = tmp_path / "flag.dat"
        flag_path.write_bytes(b"FLAG")
        run_ctx.artifacts["debug_flag"] = Artifact(
            key="debug_flag", path=flag_path, format="bin", schema="flag.v1",
            producer="test", cache_key="x", sha256="def",
        )
        key_after = compute_cache_key(proc, run_ctx)

        assert key_before == key_after
