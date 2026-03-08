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
    Map,
    OptionalInput,
    Pipeline,
    ProcessBase,
    ProducedArtifact,
    Reduce,
    RunContext,
    _build_phases,
    _ChainPhase,
    _ReducePhase,
    _SandboxedEnvironment,
    _StaticPhase,
    compute_cache_key,
    sha256_file,
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
        path = exec_ctx.out_dir / f"output_{self.model_name}.dat"
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
        path = exec_ctx.out_dir / f"final_{self.model_name}.dat"
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
        path = exec_ctx.out_dir / "summary.dat"
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
            path = exec_ctx.out_dir / f"{name}.dat"
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

    def test_valid_processes(self) -> None:
        pipeline = Pipeline([StubProcess(name="a"), StubProcess(name="b")])
        assert len(pipeline._phases) == 1


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


# ===========================================================================
# _SandboxedEnvironment
# ===========================================================================
class TestSandboxedEnvironment:
    @pytest.fixture()
    def _spy_env(self, tmp_path: Path):
        """inner env + cwd 記録用 spy を返す fixture."""
        received_cwds: list[Path | None] = []
        inner = DryRunEnvironment()
        original_run = inner.run

        def spy(command, *, cwd=None):
            received_cwds.append(cwd)
            return original_run(command, cwd=cwd)

        inner.run = spy  # type: ignore[assignment]
        return inner, received_cwds

    @dataclass(frozen=True)
    class _Echo(CommandBuilder):
        def build(self) -> list[str]:
            return ["echo", "hi"]

    def test_default_cwd_injected(self, tmp_path: Path, _spy_env) -> None:
        """cwd 未指定時にデフォルト cwd が注入される."""
        inner, received_cwds = _spy_env
        default_cwd = tmp_path / "work"
        sandbox = _SandboxedEnvironment(inner, cwd=default_cwd)

        sandbox.run(self._Echo())
        assert received_cwds == [default_cwd]

    def test_explicit_cwd_takes_precedence(self, tmp_path: Path, _spy_env) -> None:
        """明示的に cwd を渡した場合はデフォルトより優先される."""
        inner, received_cwds = _spy_env
        default_cwd = tmp_path / "default"
        explicit_cwd = tmp_path / "explicit"
        sandbox = _SandboxedEnvironment(inner, cwd=default_cwd)

        sandbox.run(self._Echo(), cwd=explicit_cwd)
        assert received_cwds == [explicit_cwd]


# ===========================================================================
# _build_phases
# ===========================================================================
class TestBuildPhases:
    def test_static_only(self) -> None:
        phases = _build_phases([StubProcess(name="a"), StubProcess(name="b")])
        assert len(phases) == 1
        assert isinstance(phases[0], _StaticPhase)
        assert len(phases[0].processes) == 2

    def test_map_chain(self) -> None:
        phases = _build_phases([Map(MappableProcess), Map(SecondMappable)])
        assert len(phases) == 1
        assert isinstance(phases[0], _ChainPhase)
        assert len(phases[0].maps) == 2

    def test_mixed_phases(self) -> None:
        phases = _build_phases([
            StubProcess(name="a"),
            Map(MappableProcess),
            Map(SecondMappable),
            Reduce(ReducibleProcess),
        ])
        assert len(phases) == 3
        assert isinstance(phases[0], _StaticPhase)
        assert isinstance(phases[1], _ChainPhase)
        assert isinstance(phases[2], _ReducePhase)


# ===========================================================================
# Map
# ===========================================================================
class TestMap:
    def test_infer_prefix(self, run_ctx: RunContext) -> None:
        m = Map(MappableProcess)
        assert m._infer_prefix() == "input"

    def test_explicit_prefix(self) -> None:
        m = Map(MappableProcess, key_prefix="custom")
        assert m._infer_prefix() == "custom"

    def test_discover_variants(self, run_ctx: RunContext, tmp_path: Path) -> None:
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

    def test_infer_prefix_fails_raises(self) -> None:
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
            m._infer_prefix()


# ===========================================================================
# Reduce
# ===========================================================================
class TestReduce:
    def test_explicit_prefix(self) -> None:
        r = Reduce(ReducibleProcess, key_prefix="custom")
        assert r._infer_prefix() == "custom"

    def test_infer_prefix_fails_raises(self) -> None:
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
            r._infer_prefix()

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
                path = ectx.out_dir / f"rec_{self.model_name}.dat"
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
        observed_cwds: list[Path | None] = []
        original_env_run = exec_ctx.env.run

        def spy_run(command, *, cwd=None):
            observed_cwds.append(cwd)
            return original_env_run(command, cwd=cwd)

        exec_ctx.env.run = spy_run  # type: ignore[assignment]

        @dataclass
        class CmdRunner(ProcessBase):
            model_name: str = "default"
            version: str = "1.0.0"

            def __post_init__(self) -> None:
                self.name = f"cmd_{self.model_name}"
                self.requires = [f"input.{self.model_name}"]
                self.produces = [f"cmd_out.{self.model_name}"]

            def run(self, ctx: RunContext, ectx: ExecContext) -> dict[str, ProducedArtifact]:
                from environment import CommandBuilder

                @dataclass(frozen=True)
                class Noop(CommandBuilder):
                    def build(self) -> list[str]:
                        return ["echo", "test"]

                ectx.env.run(Noop())
                path = ectx.out_dir / f"cmd_{self.model_name}.dat"
                path.write_bytes(b"C")
                return {f"cmd_out.{self.model_name}": ProducedArtifact(path, "bin", "v1")}

        _seed_inputs(run_ctx, tmp_path, ["a"])
        pipeline = Pipeline([Map(CmdRunner)])
        pipeline.run(run_ctx, exec_ctx)

        # _SandboxedEnvironment が inner.run() に cwd を渡す
        assert len(observed_cwds) == 1
        assert observed_cwds[0] is not None
        assert observed_cwds[0] == exec_ctx.temp_dir / "a"

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
                path = ectx.out_dir / f"output_{self.model_name}.dat"
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
                path = ectx.out_dir / f"output_{self.model_name}.dat"
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
                path = ectx.out_dir / f"output_{self.model_name}.dat"
                path.write_bytes(b"ok")
                return {f"output.{self.model_name}": ProducedArtifact(path, "bin", "v1")}

        _seed_inputs(run_ctx, tmp_path, ["a", "b"])
        pipeline = Pipeline([Map(FailOnA)])
        pipeline.run(run_ctx, exec_ctx)

        assert not (exec_ctx.temp_dir / "a").exists()
        assert not (exec_ctx.temp_dir / "b").exists()
