"""pipeline モジュール (Pipeline, compute_cache_key, RunContext, Map, Reduce) の単体テスト."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from environment import CommandBuilder, DryRunEnvironment
from typing import Any

from pipeline import (
    Artifact,
    ExecContext,
    Gate,
    Map,
    OptionalInput,
    Pipeline,
    PipelineHalted,
    ProcessBase,
    ProducedArtifact,
    Reduce,
    RunContext,
    compute_cache_key,
    sha256_file,
    sha256_path,
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
        path = exec_ctx.out_path(f"{self.name}.dat")
        path.write_bytes(self.content)
        return {"out": ProducedArtifact(path, "bin", "stub.v1")}


@dataclass
class TwoOutputProcess(ProcessBase):
    """produces が2つある Process."""

    name: str = "two_out"
    produces: list[str] = field(default_factory=lambda: ["a", "b"])
    version: str = "1.0.0"

    def run(self, ctx: RunContext, exec_ctx: ExecContext) -> dict[str, ProducedArtifact]:
        pa = exec_ctx.out_path("a.dat")
        pb = exec_ctx.out_path("b.dat")
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
        path = exec_ctx.out_path("wrong.dat")
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
        path = exec_ctx.out_path("derived.dat")
        path.write_bytes(data + b"_derived")
        return {"derived": ProducedArtifact(path, "bin", "derived.v1")}


# ---------------------------------------------------------------------------
# Map/Reduce テスト用の軽量 Process
# ---------------------------------------------------------------------------
@dataclass
class MappableProcess(ProcessBase):
    """model_name でパラメタライズされる Map 対応 Process."""

    model_name: str = "default"
    version: str = "1.0.0"
    call_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.name = f"mappable_{self.model_name}"
        self.requires = [f"input.{self.model_name}"]
        self.produces = [f"output.{self.model_name}"]

    def run(self, ctx: RunContext, exec_ctx: ExecContext) -> dict[str, ProducedArtifact]:
        self.call_count += 1
        inp = ctx.get(f"input.{self.model_name}")
        data = inp.path.read_bytes()
        path = exec_ctx.out_path(f"output_{self.model_name}.dat")
        path.write_bytes(data + b"_mapped")
        return {f"output.{self.model_name}": ProducedArtifact(path, "bin", "output.v1")}


@dataclass
class SecondMappable(ProcessBase):
    """Map チェーン用の2段目 Process."""

    model_name: str = "default"
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        self.name = f"second_{self.model_name}"
        self.requires = [f"output.{self.model_name}"]
        self.produces = [f"final.{self.model_name}"]

    def run(self, ctx: RunContext, exec_ctx: ExecContext) -> dict[str, ProducedArtifact]:
        inp = ctx.get(f"output.{self.model_name}")
        data = inp.path.read_bytes()
        path = exec_ctx.out_path(f"final_{self.model_name}.dat")
        path.write_bytes(data + b"_final")
        return {f"final.{self.model_name}": ProducedArtifact(path, "bin", "final.v1")}


@dataclass
class ReducibleProcess(ProcessBase):
    """Reduce 対応 Process."""

    model_names: list[str] = field(default_factory=list)
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        self.name = "reducer"
        self.requires = [f"output.{m}" for m in self.model_names]
        self.produces = ["summary"]

    def run(self, ctx: RunContext, exec_ctx: ExecContext) -> dict[str, ProducedArtifact]:
        parts = []
        for m in self.model_names:
            parts.append(ctx.get(f"output.{m}").path.read_bytes())
        path = exec_ctx.out_path("summary.dat")
        path.write_bytes(b"|".join(parts))
        return {"summary": ProducedArtifact(path, "bin", "summary.v1")}


@dataclass
class DynamicProducesProcess(ProcessBase):
    """動的 produces (produces=[]) の Process."""

    name: str = "dynamic"
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        self.produces = []

    call_count: int = field(default=0, init=False, repr=False)

    def run(self, ctx: RunContext, exec_ctx: ExecContext) -> dict[str, ProducedArtifact]:
        self.call_count += 1
        result = {}
        for name in ["x", "y"]:
            path = exec_ctx.out_path(f"{name}.dat")
            path.write_bytes(name.encode())
            result[f"dyn.{name}"] = ProducedArtifact(path, "bin", "dyn.v1")
        return result


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

    def test_valid_processes_runs_without_error(self, run_ctx: RunContext, exec_ctx: ExecContext) -> None:
        """有効なプロセスで構成された Pipeline は正常に実行できる."""
        pipeline = Pipeline([StubProcess(name="a"), StubProcess(name="b")])
        ctx = pipeline.run(run_ctx, exec_ctx)
        assert "out" in ctx.artifacts


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

    def test_dynamic_produces_cache_hit(self, run_ctx: RunContext, exec_ctx: ExecContext) -> None:
        proc = DynamicProducesProcess()
        pipeline = Pipeline([proc])
        pipeline.run(run_ctx, exec_ctx)
        assert proc.call_count == 1

        pipeline.run(run_ctx, exec_ctx)
        assert proc.call_count == 1

    def test_dynamic_produces_force_rerun(self, run_ctx: RunContext, exec_ctx: ExecContext) -> None:
        proc = DynamicProducesProcess()
        pipeline = Pipeline([proc])
        pipeline.run(run_ctx, exec_ctx)
        assert proc.call_count == 1

        pipeline.run(run_ctx, exec_ctx, force_processes=["dynamic"])
        assert proc.call_count == 2


# ===========================================================================
# Pipeline.run — force_processes バリデーション
# ===========================================================================
class TestPipelineRunValidation:
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
        path = exec_ctx.out_path("out.dat")
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


# ===========================================================================
# Map
# ===========================================================================
class TestMap:
    def test_discover_variants(self, run_ctx: RunContext, tmp_path: Path) -> None:
        """requires のプレフィックスを推定して variant を発見する."""
        for name in ["alpha", "beta"]:
            p = tmp_path / f"{name}.dat"
            p.write_bytes(b"X")
            run_ctx.artifacts[f"input.{name}"] = Artifact(
                key=f"input.{name}", path=p, format="bin", schema="v1",
                producer="test", cache_key="x", sha256="x",
            )

        m = Map(MappableProcess)
        variants = m.discover_variants(run_ctx)
        assert set(variants) == {"alpha", "beta"}

    def test_explicit_prefix_discovers_variants(self, run_ctx: RunContext, tmp_path: Path) -> None:
        """key_prefix を明示すると、そのプレフィックスで variant を発見する."""
        for name in ["a", "b"]:
            p = tmp_path / f"{name}.dat"
            p.write_bytes(b"X")
            run_ctx.artifacts[f"custom.{name}"] = Artifact(
                key=f"custom.{name}", path=p, format="bin", schema="v1",
                producer="test", cache_key="x", sha256="x",
            )

        m = Map(MappableProcess, key_prefix="custom")
        assert set(m.discover_variants(run_ctx)) == {"a", "b"}

    def test_expand(self, run_ctx: RunContext, tmp_path: Path) -> None:
        for name in ["a", "b"]:
            p = tmp_path / f"{name}.dat"
            p.write_bytes(b"X")
            run_ctx.artifacts[f"input.{name}"] = Artifact(
                key=f"input.{name}", path=p, format="bin", schema="v1",
                producer="test", cache_key="x", sha256="x",
            )

        m = Map(MappableProcess)
        procs = m.expand(run_ctx)
        assert len(procs) == 2
        names = {p.name for p in procs}
        assert names == {"mappable_a", "mappable_b"}

    def test_no_variants(self, run_ctx: RunContext) -> None:
        m = Map(MappableProcess)
        assert m.discover_variants(run_ctx) == []

    def test_expand_fails_when_prefix_not_inferrable(self, run_ctx: RunContext) -> None:
        """probe の requires にセンチネルが含まれない場合は ValueError."""

        @dataclass
        class NoSentinel(ProcessBase):
            model_name: str = "default"
            name: str = "no_sentinel"
            version: str = "1.0.0"

            def __post_init__(self) -> None:
                self.requires = ["fixed_key"]  # センチネルを含まない
                self.produces = [f"out.{self.model_name}"]

            def run(self, ctx: RunContext, exec_ctx: ExecContext) -> dict[str, ProducedArtifact]:
                return {}

        m = Map(NoSentinel)
        with pytest.raises(ValueError, match="Cannot infer key_prefix"):
            m.expand(run_ctx)

    def test_kwargs_factory_applied_per_variant(self, run_ctx: RunContext, tmp_path: Path) -> None:
        """kwargs_factory が variant ごとに異なるパラメータで Process を生成する."""

        @dataclass
        class ParamProcess(ProcessBase):
            model_name: str = "default"
            extra: str = ""
            version: str = "1.0.0"

            def __post_init__(self) -> None:
                self.name = f"param_{self.model_name}"
                self.requires = [f"input.{self.model_name}"]
                self.produces = [f"output.{self.model_name}"]

            def run(self, ctx: RunContext, exec_ctx: ExecContext) -> dict[str, ProducedArtifact]:
                return {}

        for name in ["a", "b"]:
            p = tmp_path / f"{name}.dat"
            p.write_bytes(b"X")
            run_ctx.artifacts[f"input.{name}"] = Artifact(
                key=f"input.{name}", path=p, format="bin", schema="v1",
                producer="test", cache_key="x", sha256="x",
            )

        m = Map(ParamProcess, kwargs_factory=lambda v: {"extra": v.upper()})
        procs = m.expand(run_ctx)

        by_name = {p.model_name: p for p in procs}
        assert by_name["a"].extra == "A"
        assert by_name["b"].extra == "B"

    def test_kwargs_and_kwargs_factory_merged_on_expand(self, run_ctx: RunContext, tmp_path: Path) -> None:
        """kwargs と kwargs_factory が両方指定された場合、マージされて Process に渡る."""

        @dataclass
        class MultiParamProcess(ProcessBase):
            model_name: str = "default"
            static: int = 0
            dynamic: str = ""
            version: str = "1.0.0"

            def __post_init__(self) -> None:
                self.name = f"mp_{self.model_name}"
                self.requires = [f"input.{self.model_name}"]
                self.produces = [f"output.{self.model_name}"]

            def run(self, ctx: RunContext, exec_ctx: ExecContext) -> dict[str, ProducedArtifact]:
                return {}

        p = tmp_path / "x.dat"
        p.write_bytes(b"X")
        run_ctx.artifacts["input.x"] = Artifact(
            key="input.x", path=p, format="bin", schema="v1",
            producer="test", cache_key="x", sha256="x",
        )

        m = Map(
            MultiParamProcess,
            kwargs={"static": 1},
            kwargs_factory=lambda v: {"dynamic": v},
        )
        procs = m.expand(run_ctx)
        assert len(procs) == 1
        assert procs[0].static == 1
        assert procs[0].dynamic == "x"


# ===========================================================================
# Reduce
# ===========================================================================
class TestReduce:
    def test_explicit_prefix_discovers_variants(self, run_ctx: RunContext, tmp_path: Path) -> None:
        """key_prefix を明示すると、そのプレフィックスで variant を発見する."""
        for name in ["a", "b"]:
            p = tmp_path / f"{name}.dat"
            p.write_bytes(b"X")
            run_ctx.artifacts[f"custom.{name}"] = Artifact(
                key=f"custom.{name}", path=p, format="bin", schema="v1",
                producer="test", cache_key="x", sha256="x",
            )

        r = Reduce(ReducibleProcess, key_prefix="custom")
        assert set(r.discover_variants(run_ctx)) == {"a", "b"}

    def test_expand_fails_when_prefix_not_inferrable(self, run_ctx: RunContext) -> None:
        @dataclass
        class NoSentinelReduce(ProcessBase):
            model_names: list[str] = field(default_factory=list)
            name: str = "no_sentinel_r"
            version: str = "1.0.0"

            def __post_init__(self) -> None:
                self.requires = ["fixed_key"]
                self.produces = ["summary"]

            def run(self, ctx: RunContext, exec_ctx: ExecContext) -> dict[str, ProducedArtifact]:
                return {}

        r = Reduce(NoSentinelReduce)
        with pytest.raises(ValueError, match="Cannot infer key_prefix"):
            r.expand(run_ctx)

    def test_expand(self, run_ctx: RunContext, tmp_path: Path) -> None:
        for name in ["a", "b"]:
            p = tmp_path / f"{name}.dat"
            p.write_bytes(b"X")
            run_ctx.artifacts[f"output.{name}"] = Artifact(
                key=f"output.{name}", path=p, format="bin", schema="v1",
                producer="test", cache_key="x", sha256="x",
            )

        r = Reduce(ReducibleProcess)
        proc = r.expand(run_ctx)
        assert proc.name == "reducer"
        assert set(proc.requires) == {"output.a", "output.b"}


# ===========================================================================
# Pipeline — Map/Reduce E2E
# ===========================================================================
def _seed_inputs(run_ctx: RunContext, tmp_path: Path, names: list[str]) -> None:
    """テスト用に input.<name> アーティファクトを事前登録."""
    for name in names:
        p = tmp_path / f"input_{name}.dat"
        p.write_bytes(name.encode())
        run_ctx.put(Artifact(
            key=f"input.{name}", path=p, format="bin", schema="v1",
            producer="seed", cache_key="seed",
            sha256=sha256_file(p),
        ))


class TestPipelineMapReduce:
    def test_single_map(self, run_ctx: RunContext, exec_ctx: ExecContext, tmp_path: Path) -> None:
        _seed_inputs(run_ctx, tmp_path, ["a", "b"])

        pipeline = Pipeline([Map(MappableProcess)])
        ctx = pipeline.run(run_ctx, exec_ctx)

        assert "output.a" in ctx.artifacts
        assert "output.b" in ctx.artifacts
        assert ctx.artifacts["output.a"].path.read_bytes() == b"a_mapped"
        assert ctx.artifacts["output.b"].path.read_bytes() == b"b_mapped"

    def test_chained_maps(self, run_ctx: RunContext, exec_ctx: ExecContext, tmp_path: Path) -> None:
        _seed_inputs(run_ctx, tmp_path, ["x", "y"])

        pipeline = Pipeline([Map(MappableProcess), Map(SecondMappable)])
        ctx = pipeline.run(run_ctx, exec_ctx)

        assert "final.x" in ctx.artifacts
        assert "final.y" in ctx.artifacts
        assert ctx.artifacts["final.x"].path.read_bytes() == b"x_mapped_final"

    def test_map_then_reduce(self, run_ctx: RunContext, exec_ctx: ExecContext, tmp_path: Path) -> None:
        _seed_inputs(run_ctx, tmp_path, ["a", "b"])

        pipeline = Pipeline([
            Map(MappableProcess),
            Reduce(ReducibleProcess),
        ])
        ctx = pipeline.run(run_ctx, exec_ctx)

        assert "summary" in ctx.artifacts
        summary = ctx.artifacts["summary"].path.read_bytes()
        assert b"a_mapped" in summary
        assert b"b_mapped" in summary

    def test_empty_variants_skips(self, run_ctx: RunContext, exec_ctx: ExecContext) -> None:
        pipeline = Pipeline([Map(MappableProcess)])
        ctx = pipeline.run(run_ctx, exec_ctx)
        # variant が 0 個 → 何も起きないが例外にならない
        assert len([k for k in ctx.artifacts if k.startswith("output.")]) == 0

    def test_chain_uses_separate_temp_dirs(
        self, run_ctx: RunContext, exec_ctx: ExecContext, tmp_path: Path,
    ) -> None:
        """各 variant チェーンに固有の temp_dir が割り当てられることを確認."""
        observed_temp_dirs: list[Path] = []

        @dataclass
        class TempRecorder(ProcessBase):
            model_name: str = "default"
            version: str = "1.0.0"

            def __post_init__(self) -> None:
                self.name = f"rec_{self.model_name}"
                self.requires = [f"input.{self.model_name}"]
                self.produces = [f"rec_out.{self.model_name}"]

            def run(self, ctx: RunContext, ectx: ExecContext) -> dict[str, ProducedArtifact]:
                observed_temp_dirs.append(ectx.temp_dir)
                path = ectx.out_path(f"rec_{self.model_name}.dat")
                path.write_bytes(b"R")
                return {f"rec_out.{self.model_name}": ProducedArtifact(path, "bin", "v1")}

        _seed_inputs(run_ctx, tmp_path, ["a", "b"])
        pipeline = Pipeline([Map(TempRecorder)])
        pipeline.run(run_ctx, exec_ctx)

        assert len(observed_temp_dirs) == 2
        assert observed_temp_dirs[0] != observed_temp_dirs[1]

    def test_sandboxed_env_injects_cwd(
        self, run_ctx: RunContext, exec_ctx: ExecContext, tmp_path: Path,
    ) -> None:
        """Map 内の env.run() にデフォルト cwd が注入されることを確認."""
        env = DryRunEnvironment()
        exec_ctx = ExecContext(
            out_dir=exec_ctx.out_dir, temp_dir=exec_ctx.temp_dir,
            logger=exec_ctx.logger, env=env,
        )

        @dataclass
        class CmdRunner(ProcessBase):
            model_name: str = "default"
            version: str = "1.0.0"

            def __post_init__(self) -> None:
                self.name = f"cmd_{self.model_name}"
                self.requires = [f"input.{self.model_name}"]
                self.produces = [f"cmd_out.{self.model_name}"]

            def run(self, ctx: RunContext, ectx: ExecContext) -> dict[str, ProducedArtifact]:
                @dataclass(frozen=True)
                class Noop(CommandBuilder):
                    def build(self) -> list[str]:
                        return ["echo", "test"]

                ectx.env.run(Noop())
                path = ectx.out_path(f"cmd_{self.model_name}.dat")
                path.write_bytes(b"C")
                return {f"cmd_out.{self.model_name}": ProducedArtifact(path, "bin", "v1")}

        _seed_inputs(run_ctx, tmp_path, ["a"])
        pipeline = Pipeline([Map(CmdRunner)])
        pipeline.run(run_ctx, exec_ctx)

        # DryRunEnvironment.records で cwd を直接確認
        assert len(env.records) == 1
        assert env.records[0].cwd == exec_ctx.temp_dir / "a"

    def test_artifacts_relocated_to_out_dir(
        self, run_ctx: RunContext, exec_ctx: ExecContext, tmp_path: Path,
    ) -> None:
        """Map で生成されたアーティファクトが out_dir に移動されることを確認."""
        _seed_inputs(run_ctx, tmp_path, ["a", "b"])

        pipeline = Pipeline([Map(MappableProcess)])
        ctx = pipeline.run(run_ctx, exec_ctx)

        for name in ["a", "b"]:
            art = ctx.artifacts[f"output.{name}"]
            assert art.path.parent == exec_ctx.out_dir
            assert art.path.exists()

    def test_temp_dirs_cleaned_up(
        self, run_ctx: RunContext, exec_ctx: ExecContext, tmp_path: Path,
    ) -> None:
        """Map 完了後に一時ディレクトリが削除されることを確認."""
        _seed_inputs(run_ctx, tmp_path, ["a", "b"])

        pipeline = Pipeline([Map(MappableProcess)])
        pipeline.run(run_ctx, exec_ctx)

        # variant ごとの一時ディレクトリが消えている
        assert not (exec_ctx.temp_dir / "a").exists()
        assert not (exec_ctx.temp_dir / "b").exists()

    def test_partial_failure_continues(
        self, run_ctx: RunContext, exec_ctx: ExecContext, tmp_path: Path,
    ) -> None:
        """一部 variant が失敗しても他の variant は完走する."""

        @dataclass
        class FailOnA(ProcessBase):
            model_name: str = "default"
            version: str = "1.0.0"

            def __post_init__(self) -> None:
                self.name = f"fail_on_a_{self.model_name}"
                self.requires = [f"input.{self.model_name}"]
                self.produces = [f"output.{self.model_name}"]

            def run(self, ctx: RunContext, ectx: ExecContext) -> dict[str, ProducedArtifact]:
                if self.model_name == "a":
                    raise RuntimeError("Intentional failure for 'a'")
                inp = ctx.get(f"input.{self.model_name}")
                path = ectx.out_path(f"output_{self.model_name}.dat")
                path.write_bytes(inp.path.read_bytes() + b"_ok")
                return {f"output.{self.model_name}": ProducedArtifact(path, "bin", "v1")}

        _seed_inputs(run_ctx, tmp_path, ["a", "b"])
        pipeline = Pipeline([Map(FailOnA)])
        ctx = pipeline.run(run_ctx, exec_ctx)

        # "a" は失敗 → artifact なし
        assert "output.a" not in ctx.artifacts
        # "b" は成功
        assert "output.b" in ctx.artifacts
        assert ctx.artifacts["output.b"].path.exists()

    def test_partial_failure_reduce_uses_survivors(
        self, run_ctx: RunContext, exec_ctx: ExecContext, tmp_path: Path,
    ) -> None:
        """失敗した variant を除いて Reduce が実行される."""

        @dataclass
        class FailOnA(ProcessBase):
            model_name: str = "default"
            version: str = "1.0.0"

            def __post_init__(self) -> None:
                self.name = f"fail_on_a_{self.model_name}"
                self.requires = [f"input.{self.model_name}"]
                self.produces = [f"output.{self.model_name}"]

            def run(self, ctx: RunContext, ectx: ExecContext) -> dict[str, ProducedArtifact]:
                if self.model_name == "a":
                    raise RuntimeError("Intentional failure for 'a'")
                inp = ctx.get(f"input.{self.model_name}")
                path = ectx.out_path(f"output_{self.model_name}.dat")
                path.write_bytes(inp.path.read_bytes() + b"_ok")
                return {f"output.{self.model_name}": ProducedArtifact(path, "bin", "v1")}

        _seed_inputs(run_ctx, tmp_path, ["a", "b"])
        pipeline = Pipeline([Map(FailOnA), Reduce(ReducibleProcess)])
        ctx = pipeline.run(run_ctx, exec_ctx)

        # Reduce は "b" のみで実行される
        assert "summary" in ctx.artifacts
        summary = ctx.artifacts["summary"].path.read_bytes()
        assert b"b_ok" in summary
        assert b"a" not in summary

    def test_partial_failure_cleans_temp_dirs(
        self, run_ctx: RunContext, exec_ctx: ExecContext, tmp_path: Path,
    ) -> None:
        """失敗した variant の一時ディレクトリも削除される."""

        @dataclass
        class FailOnA(ProcessBase):
            model_name: str = "default"
            version: str = "1.0.0"

            def __post_init__(self) -> None:
                self.name = f"fail_on_a_{self.model_name}"
                self.requires = [f"input.{self.model_name}"]
                self.produces = [f"output.{self.model_name}"]

            def run(self, ctx: RunContext, ectx: ExecContext) -> dict[str, ProducedArtifact]:
                if self.model_name == "a":
                    raise RuntimeError("Intentional failure for 'a'")
                path = ectx.out_path(f"output_{self.model_name}.dat")
                path.write_bytes(b"ok")
                return {f"output.{self.model_name}": ProducedArtifact(path, "bin", "v1")}

        _seed_inputs(run_ctx, tmp_path, ["a", "b"])
        pipeline = Pipeline([Map(FailOnA)])
        pipeline.run(run_ctx, exec_ctx)

        assert not (exec_ctx.temp_dir / "a").exists()
        assert not (exec_ctx.temp_dir / "b").exists()


# ===========================================================================
# Pipeline — skip_if_missing
# ===========================================================================
@dataclass
class SkippableProcess(ProcessBase):
    """skip_if_missing=True の Process."""

    name: str = "skippable"
    requires: list[str] = field(default_factory=lambda: ["maybe_missing"])
    produces: list[str] = field(default_factory=lambda: ["skippable_out"])
    version: str = "1.0.0"
    skip_if_missing: bool = True

    call_count: int = field(default=0, init=False, repr=False)

    def run(self, ctx: RunContext, exec_ctx: ExecContext) -> dict[str, ProducedArtifact]:
        self.call_count += 1
        path = exec_ctx.out_path("skippable.dat")
        path.write_bytes(b"RAN")
        return {"skippable_out": ProducedArtifact(path, "bin", "skip.v1")}


class TestSkipIfMissing:
    def test_skips_when_artifact_missing(self, run_ctx: RunContext, exec_ctx: ExecContext) -> None:
        """requires のアーティファクトが無ければスキップされる."""
        proc = SkippableProcess()
        pipeline = Pipeline([proc])
        ctx = pipeline.run(run_ctx, exec_ctx)

        assert proc.call_count == 0
        assert "skippable_out" not in ctx.artifacts

    def test_runs_when_artifact_present(
        self, run_ctx: RunContext, exec_ctx: ExecContext, tmp_path: Path,
    ) -> None:
        """requires のアーティファクトがあれば通常通り実行される."""
        p = tmp_path / "input.dat"
        p.write_bytes(b"DATA")
        run_ctx.put(Artifact(
            key="maybe_missing", path=p, format="bin", schema="v1",
            producer="test", cache_key="x", sha256=sha256_file(p),
        ))

        proc = SkippableProcess()
        pipeline = Pipeline([proc])
        ctx = pipeline.run(run_ctx, exec_ctx)

        assert proc.call_count == 1
        assert "skippable_out" in ctx.artifacts

    def test_downstream_unaffected_by_skip(self, run_ctx: RunContext, exec_ctx: ExecContext) -> None:
        """skip されたプロセスの後続は requires が満たされていれば実行される."""
        pipeline = Pipeline([
            StubProcess(),
            SkippableProcess(),
            DependentProcess(),  # requires=["out"] → StubProcess の出力
        ])
        ctx = pipeline.run(run_ctx, exec_ctx)

        assert "out" in ctx.artifacts
        assert "skippable_out" not in ctx.artifacts
        assert "derived" in ctx.artifacts

    def test_skip_if_missing_false_raises_on_missing(self, run_ctx: RunContext, exec_ctx: ExecContext) -> None:
        """skip_if_missing=False (デフォルト) ならアーティファクト不在で KeyError."""
        proc = StubProcess(name="needs_input", requires=["nonexistent"])
        pipeline = Pipeline([proc])
        with pytest.raises(KeyError, match="Missing artifact"):
            pipeline.run(run_ctx, exec_ctx)


# ===========================================================================
# Gate / PipelineHalted
# ===========================================================================
class TestGate:
    def test_gate_passes(self, run_ctx: RunContext, exec_ctx: ExecContext) -> None:
        """check が True → Gate を通過して後続が実行される."""
        pipeline = Pipeline([
            StubProcess(),
            Gate(check=lambda ctx: True),
            DependentProcess(),
        ])
        ctx = pipeline.run(run_ctx, exec_ctx)
        assert "derived" in ctx.artifacts

    def test_gate_halts(self, run_ctx: RunContext, exec_ctx: ExecContext) -> None:
        """check が False → PipelineHalted が送出される."""
        pipeline = Pipeline([
            StubProcess(),
            Gate(check=lambda ctx: False, message="Recipe not ready"),
            DependentProcess(),
        ])
        with pytest.raises(PipelineHalted, match="Recipe not ready") as exc_info:
            pipeline.run(run_ctx, exec_ctx)

        # Gate 前のプロセスは実行済み
        assert "out" in exc_info.value.ctx.artifacts
        # Gate 後のプロセスは未実行
        assert "derived" not in exc_info.value.ctx.artifacts

    def test_gate_resume_after_condition_met(
        self, run_ctx: RunContext, exec_ctx: ExecContext,
    ) -> None:
        """1回目は停止、条件を満たして2回目は通過する."""
        ready = {"value": False}

        pipeline = Pipeline([
            StubProcess(),
            Gate(check=lambda ctx: ready["value"]),
            DependentProcess(),
        ])

        # 1回目: Gate で停止
        with pytest.raises(PipelineHalted):
            pipeline.run(run_ctx, exec_ctx)
        assert "out" in run_ctx.artifacts
        assert "derived" not in run_ctx.artifacts

        # 条件を満たす
        ready["value"] = True

        # 2回目: StubProcess はキャッシュヒット、Gate 通過、DependentProcess 実行
        ctx = pipeline.run(run_ctx, exec_ctx)
        assert "derived" in ctx.artifacts

    def test_gate_between_static_and_map(
        self, run_ctx: RunContext, exec_ctx: ExecContext, tmp_path: Path,
    ) -> None:
        """Static → Gate → Map の構成で正しく実行される."""
        pipeline = Pipeline([
            StubProcess(),
            Gate(check=lambda ctx: True),
            Map(MappableProcess),
        ])

        # Map が input.* を発見するためのアーティファクトを登録
        p = tmp_path / "a.dat"
        p.write_bytes(b"X")
        run_ctx.artifacts["input.a"] = Artifact(
            key="input.a", path=p, format="bin", schema="v1",
            producer="test", cache_key="x", sha256="x",
        )

        ctx = pipeline.run(run_ctx, exec_ctx)
        assert "out" in ctx.artifacts
        assert "output.a" in ctx.artifacts


# ===========================================================================
# Pipeline — per-process Environment
# ===========================================================================
class TestPerProcessEnv:
    def test_static_process_uses_own_env(
        self, run_ctx: RunContext, exec_ctx: ExecContext,
    ) -> None:
        """ProcessBase.env が設定されていれば、そちらが使用される."""
        proc_env = DryRunEnvironment()

        @dataclass
        class EnvProcess(ProcessBase):
            name: str = "env_proc"
            produces: list[str] = field(default_factory=lambda: ["env_out"])
            version: str = "1.0.0"

            def run(self, ctx: RunContext, ectx: ExecContext) -> dict[str, ProducedArtifact]:
                @dataclass(frozen=True)
                class Noop(CommandBuilder):
                    def build(self) -> list[str]:
                        return ["echo", "hello"]

                ectx.env.run(Noop())
                path = ectx.out_path("env_out.dat")
                path.write_bytes(b"ENV")
                return {"env_out": ProducedArtifact(path, "bin", "v1")}

        proc = EnvProcess(env=proc_env)
        pipeline = Pipeline([proc])
        pipeline.run(run_ctx, exec_ctx)

        # proc_env にコマンドが記録される
        assert len(proc_env.records) == 1
        assert proc_env.records[0].command.build() == ["echo", "hello"]

    def test_map_env_overrides_default(
        self, run_ctx: RunContext, exec_ctx: ExecContext, tmp_path: Path,
    ) -> None:
        """Map.env が設定されていれば、その Map のプロセスはそちらの env を使う."""
        map_env = DryRunEnvironment()

        @dataclass
        class CmdMappable(ProcessBase):
            model_name: str = "default"
            version: str = "1.0.0"

            def __post_init__(self) -> None:
                self.name = f"cmd_map_{self.model_name}"
                self.requires = [f"input.{self.model_name}"]
                self.produces = [f"output.{self.model_name}"]

            def run(self, ctx: RunContext, ectx: ExecContext) -> dict[str, ProducedArtifact]:
                @dataclass(frozen=True)
                class Noop(CommandBuilder):
                    def build(self) -> list[str]:
                        return ["run", self.model_name]

                    model_name: str = "default"

                ectx.env.run(Noop(model_name=self.model_name))
                path = ectx.out_path(f"output_{self.model_name}.dat")
                path.write_bytes(b"OK")
                return {f"output.{self.model_name}": ProducedArtifact(path, "bin", "v1")}

        _seed_inputs(run_ctx, tmp_path, ["a", "b"])
        pipeline = Pipeline([Map(CmdMappable, env=map_env)])
        pipeline.run(run_ctx, exec_ctx)

        # map_env に 2 variant 分のコマンドが記録される
        assert len(map_env.records) == 2
        cmds = {tuple(r.command.build()) for r in map_env.records}
        assert ("run", "a") in cmds
        assert ("run", "b") in cmds

    def test_mixed_chain_env(
        self, run_ctx: RunContext, exec_ctx: ExecContext, tmp_path: Path,
    ) -> None:
        """連続 Map で一部だけ env を指定すると、指定した Map のみ別 env を使う."""
        special_env = DryRunEnvironment()
        default_env = DryRunEnvironment()
        exec_ctx = ExecContext(
            out_dir=exec_ctx.out_dir, temp_dir=exec_ctx.temp_dir,
            logger=exec_ctx.logger, env=default_env,
        )

        @dataclass
        class Step1(ProcessBase):
            model_name: str = "default"
            version: str = "1.0.0"

            def __post_init__(self) -> None:
                self.name = f"step1_{self.model_name}"
                self.requires = [f"input.{self.model_name}"]
                self.produces = [f"mid.{self.model_name}"]

            def run(self, ctx: RunContext, ectx: ExecContext) -> dict[str, ProducedArtifact]:
                @dataclass(frozen=True)
                class Cmd1(CommandBuilder):
                    def build(self) -> list[str]:
                        return ["step1"]

                ectx.env.run(Cmd1())
                path = ectx.out_path(f"mid_{self.model_name}.dat")
                path.write_bytes(b"MID")
                return {f"mid.{self.model_name}": ProducedArtifact(path, "bin", "v1")}

        @dataclass
        class Step2(ProcessBase):
            model_name: str = "default"
            version: str = "1.0.0"

            def __post_init__(self) -> None:
                self.name = f"step2_{self.model_name}"
                self.requires = [f"mid.{self.model_name}"]
                self.produces = [f"output.{self.model_name}"]

            def run(self, ctx: RunContext, ectx: ExecContext) -> dict[str, ProducedArtifact]:
                @dataclass(frozen=True)
                class Cmd2(CommandBuilder):
                    def build(self) -> list[str]:
                        return ["step2"]

                ectx.env.run(Cmd2())
                path = ectx.out_path(f"output_{self.model_name}.dat")
                path.write_bytes(b"OUT")
                return {f"output.{self.model_name}": ProducedArtifact(path, "bin", "v1")}

        _seed_inputs(run_ctx, tmp_path, ["a"])
        pipeline = Pipeline([
            Map(Step1),                      # デフォルト env
            Map(Step2, env=special_env),     # 専用 env
        ])
        pipeline.run(run_ctx, exec_ctx)

        # Step1 はデフォルト env (SandboxedEnvironment 経由で default_env に記録)
        assert len(default_env.records) == 1
        assert default_env.records[0].command.build() == ["step1"]
        # Step2 は special_env
        assert len(special_env.records) == 1
        assert special_env.records[0].command.build() == ["step2"]


# ===========================================================================
# sha256_path — ファイル / ディレクトリ 両対応
# ===========================================================================
class TestSha256Path:
    def test_file_matches_sha256_file(self, tmp_path: Path) -> None:
        """単一ファイルの場合、sha256_file と同じ結果を返す."""
        f = tmp_path / "a.txt"
        f.write_bytes(b"hello")
        assert sha256_path(f) == sha256_file(f)

    def test_directory_hash(self, tmp_path: Path) -> None:
        """ディレクトリのハッシュが計算できる."""
        d = tmp_path / "dir"
        d.mkdir()
        (d / "a.txt").write_bytes(b"AAA")
        (d / "b.txt").write_bytes(b"BBB")
        h = sha256_path(d)
        assert isinstance(h, str) and len(h) == 64

    def test_directory_hash_deterministic(self, tmp_path: Path) -> None:
        """同じ内容のディレクトリは同じハッシュを返す."""
        d = tmp_path / "dir"
        d.mkdir()
        (d / "x.txt").write_bytes(b"X")
        (d / "y.txt").write_bytes(b"Y")
        assert sha256_path(d) == sha256_path(d)

    def test_directory_content_change_changes_hash(self, tmp_path: Path) -> None:
        """ファイル内容が変わるとハッシュが変わる."""
        d = tmp_path / "dir"
        d.mkdir()
        (d / "a.txt").write_bytes(b"V1")
        h1 = sha256_path(d)
        (d / "a.txt").write_bytes(b"V2")
        h2 = sha256_path(d)
        assert h1 != h2

    def test_directory_file_added_changes_hash(self, tmp_path: Path) -> None:
        """ファイルが追加されるとハッシュが変わる."""
        d = tmp_path / "dir"
        d.mkdir()
        (d / "a.txt").write_bytes(b"A")
        h1 = sha256_path(d)
        (d / "b.txt").write_bytes(b"B")
        h2 = sha256_path(d)
        assert h1 != h2

    def test_directory_nested(self, tmp_path: Path) -> None:
        """ネストされたディレクトリも走査される."""
        d = tmp_path / "dir"
        (d / "sub").mkdir(parents=True)
        (d / "top.txt").write_bytes(b"TOP")
        (d / "sub" / "inner.txt").write_bytes(b"INNER")
        h = sha256_path(d)
        assert isinstance(h, str) and len(h) == 64

    def test_empty_directory(self, tmp_path: Path) -> None:
        """空ディレクトリでもハッシュが計算できる."""
        d = tmp_path / "empty"
        d.mkdir()
        h = sha256_path(d)
        assert isinstance(h, str) and len(h) == 64


# ===========================================================================
# Pipeline — ディレクトリ Artifact
# ===========================================================================
@dataclass
class DirProducerProcess(ProcessBase):
    """ディレクトリを生成物とする Process."""

    name: str = "dir_producer"
    produces: list[str] = field(default_factory=lambda: ["dir_out"])
    version: str = "1.0.0"

    call_count: int = field(default=0, init=False, repr=False)

    def run(self, ctx: RunContext, exec_ctx: ExecContext) -> dict[str, ProducedArtifact]:
        self.call_count += 1
        d = exec_ctx.out_dir / "output_dir"
        d.mkdir(parents=True, exist_ok=True)
        (d / "a.txt").write_bytes(b"AAA")
        (d / "b.txt").write_bytes(b"BBB")
        return {"dir_out": ProducedArtifact(d, "directory", "dir.v1")}


class TestDirectoryArtifact:
    def test_directory_artifact_registered(self, run_ctx: RunContext, exec_ctx: ExecContext) -> None:
        """ディレクトリを生成物として登録できる."""
        pipeline = Pipeline([DirProducerProcess()])
        ctx = pipeline.run(run_ctx, exec_ctx)

        assert "dir_out" in ctx.artifacts
        art = ctx.artifacts["dir_out"]
        assert art.path.is_dir()
        assert art.format == "directory"
        assert len(art.sha256) == 64

    def test_directory_artifact_cache_hit(self, run_ctx: RunContext, exec_ctx: ExecContext) -> None:
        """ディレクトリ Artifact もキャッシュが効く."""
        proc = DirProducerProcess()
        pipeline = Pipeline([proc])
        pipeline.run(run_ctx, exec_ctx)
        assert proc.call_count == 1

        pipeline.run(run_ctx, exec_ctx)
        assert proc.call_count == 1  # キャッシュヒット

    def test_directory_artifact_cache_invalidated_on_change(
        self, run_ctx: RunContext, exec_ctx: ExecContext,
    ) -> None:
        """ディレクトリ内のファイルが変更されるとキャッシュが無効化される."""
        proc = DirProducerProcess()
        pipeline = Pipeline([proc])
        pipeline.run(run_ctx, exec_ctx)
        assert proc.call_count == 1

        # ディレクトリ内のファイルを書き換え
        art = run_ctx.artifacts["dir_out"]
        (art.path / "a.txt").write_bytes(b"CHANGED")

        pipeline.run(run_ctx, exec_ctx)
        assert proc.call_count == 2  # キャッシュミス → 再実行


# ===========================================================================
# Pipeline — allow_failure
# ===========================================================================
@dataclass
class FailingProcess(ProcessBase):
    """常に例外を送出する Process."""

    name: str = "failing"
    produces: list[str] = field(default_factory=lambda: ["fail_out"])
    version: str = "1.0.0"
    allow_failure: bool = True

    def run(self, ctx: RunContext, exec_ctx: ExecContext) -> dict[str, ProducedArtifact]:
        raise RuntimeError("Intentional failure")


@dataclass
class PartialOutputProcess(ProcessBase):
    """produces を宣言するが空 dict を返す Process."""

    name: str = "partial"
    produces: list[str] = field(default_factory=lambda: ["partial_out"])
    version: str = "1.0.0"
    allow_failure: bool = True

    call_count: int = field(default=0, init=False, repr=False)

    def run(self, ctx: RunContext, exec_ctx: ExecContext) -> dict[str, ProducedArtifact]:
        self.call_count += 1
        return {}


class TestAllowFailure:
    def test_exception_caught_and_pipeline_continues(
        self, run_ctx: RunContext, exec_ctx: ExecContext,
    ) -> None:
        """allow_failure=True なら例外を送出してもパイプラインが続行する."""
        pipeline = Pipeline([
            StubProcess(),
            FailingProcess(),
            DependentProcess(),  # requires=["out"] → StubProcess の出力
        ])
        ctx = pipeline.run(run_ctx, exec_ctx)

        assert "out" in ctx.artifacts
        assert "fail_out" not in ctx.artifacts
        assert "derived" in ctx.artifacts

    def test_partial_output_accepted(
        self, run_ctx: RunContext, exec_ctx: ExecContext,
    ) -> None:
        """allow_failure=True なら produces と不一致でもエラーにならない."""
        # PartialOutputProcess は常に {} を返すため、DryRun ではファントムが
        # 登録されてしまう → LocalEnvironment を使う
        from environment import LocalEnvironment
        local_exec_ctx = ExecContext(
            out_dir=exec_ctx.out_dir, temp_dir=exec_ctx.temp_dir,
            logger=exec_ctx.logger, env=LocalEnvironment(),
        )
        proc = PartialOutputProcess()
        pipeline = Pipeline([proc])
        ctx = pipeline.run(run_ctx, local_exec_ctx)

        assert proc.call_count == 1
        assert "partial_out" not in ctx.artifacts

    def test_partial_output_no_cache(
        self, run_ctx: RunContext, exec_ctx: ExecContext,
    ) -> None:
        """空出力はキャッシュされず、再実行時にも再度 run() が呼ばれる."""
        from environment import LocalEnvironment
        local_exec_ctx = ExecContext(
            out_dir=exec_ctx.out_dir, temp_dir=exec_ctx.temp_dir,
            logger=exec_ctx.logger, env=LocalEnvironment(),
        )
        proc = PartialOutputProcess()
        pipeline = Pipeline([proc])
        pipeline.run(run_ctx, local_exec_ctx)
        assert proc.call_count == 1

        pipeline.run(run_ctx, local_exec_ctx)
        assert proc.call_count == 2  # キャッシュなし → 再実行

    def test_allow_failure_false_raises_on_exception(
        self, run_ctx: RunContext, exec_ctx: ExecContext,
    ) -> None:
        """allow_failure=False (デフォルト) なら例外は伝播する."""
        proc = FailingProcess(allow_failure=False)
        pipeline = Pipeline([proc])
        with pytest.raises(RuntimeError, match="Intentional failure"):
            pipeline.run(run_ctx, exec_ctx)

    def test_allow_failure_false_raises_on_mismatch(
        self, run_ctx: RunContext, exec_ctx: ExecContext,
    ) -> None:
        """allow_failure=False なら produces 不一致でエラーになる."""
        # PartialOutputProcess は常に {} を返すため、DryRun では
        # 「DryRun で実行できなかった」と区別がつかない → LocalEnvironment を使う
        from environment import LocalEnvironment
        local_exec_ctx = ExecContext(
            out_dir=exec_ctx.out_dir, temp_dir=exec_ctx.temp_dir,
            logger=exec_ctx.logger, env=LocalEnvironment(),
        )
        proc = PartialOutputProcess(allow_failure=False)
        pipeline = Pipeline([proc])
        with pytest.raises(RuntimeError, match="produces mismatch"):
            pipeline.run(run_ctx, local_exec_ctx)


# ===========================================================================
# Environment.executes
# ===========================================================================
class TestEnvironmentExecutes:
    def test_dry_run_does_not_execute(self) -> None:
        env = DryRunEnvironment()
        assert env.executes is False

    def test_default_executes(self) -> None:
        """Environment のデフォルトは True."""
        from environment import LocalEnvironment
        env = LocalEnvironment()
        assert env.executes is True

    def test_sandboxed_delegates_to_inner(self, tmp_path: Path) -> None:
        """_SandboxedEnvironment は inner の executes を返す."""
        from pipeline import _SandboxedEnvironment

        dry = DryRunEnvironment()
        sandboxed = _SandboxedEnvironment(dry, cwd=tmp_path)
        assert sandboxed.executes is False
