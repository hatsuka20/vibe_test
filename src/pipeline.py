import hashlib
import json
import logging
import shutil
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from environment import CommandBuilder, CommandResult, Environment, LocalEnvironment


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_json_dumps(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")

@dataclass(frozen=True)
class Artifact:
    key: str
    path: Path
    format: str
    schema: str
    producer: str
    cache_key: str
    sha256: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OptionalInput:
    key: str
    affects_cache: bool = True


@dataclass
class RunContext:
    run_dir: Path  # NOTE: 意味論的にはExecContextに寄せたいが、参照解決や実装の複雑化等からここに配置
    artifacts: dict[str, Artifact] = field(default_factory=dict)

    manifest_path: Path = field(init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.manifest_path = self.run_dir / "manifest.json"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Artifact:
        with self._lock:
            if key not in self.artifacts:
                raise KeyError(f"Missing artifact: {key}")
            return self.artifacts[key]

    def get_optional(self, key: str) -> Artifact | None:
        with self._lock:
            return self.artifacts.get(key)

    def put(self, art: Artifact) -> None:
        with self._lock:
            self.artifacts[art.key] = art
            self.save_manifest()

    def save_manifest(self) -> None:
        def _serialize(v: Artifact) -> dict[str, Any]:
            d = asdict(v)
            d["path"] = str(d["path"])
            return d

        obj = {
            "run_dir": str(self.run_dir),
            "artifacts": {k: _serialize(v) for k, v in self.artifacts.items()},
        }
        json_dump(self.manifest_path, obj)

    @classmethod
    def load(cls, run_dir: Path) -> "RunContext":
        ctx = cls(run_dir=run_dir)
        mp = ctx.manifest_path
        if mp.exists():
            obj = json_load(mp)
            arts = {}
            for k, v in obj.get("artifacts", {}).items():
                v["path"] = Path(v["path"])
                arts[k] = Artifact(**v)
            ctx.artifacts = arts
        return ctx


class _SandboxedEnvironment(Environment):
    """全コマンドにデフォルト cwd を注入する Environment ラッパー."""

    def __init__(self, inner: Environment, cwd: Path) -> None:
        self._inner = inner
        self._cwd = cwd

    def run(self, command: CommandBuilder, *, cwd: Path | None = None) -> CommandResult:
        return self._inner.run(command, cwd=cwd or self._cwd)


@dataclass
class ExecContext:
    out_dir: Path
    temp_dir: Path
    logger: logging.Logger
    env: Environment = field(default_factory=LocalEnvironment)


@dataclass(frozen=True)
class ProducedArtifact:
    path: Path
    format: str
    schema: str


@dataclass
class ProcessBase(ABC):
    """Process の基底クラス. 共通フィールドのデフォルトを提供する.

    Versioning rules:
        version — Process 実装のバージョン (semver).
            同じ入力に対して出力の「値」が変わりうる変更で bump する.
            キャッシュ無効化のトリガーとして使用される.
        ProducedArtifact.schema — 出力の構造的契約.
            下流 Process が期待するフォーマット仕様を表す.
            出力構造の破壊的変更でのみ更新する.
    両者は独立した軸であり、version の bump が schema の変更を
    必ずしも伴わない (例: バグ修正)."""

    name: str = ""
    version: str = "0.0.0"
    requires: list[str] = field(default_factory=list)
    optional: list[OptionalInput] = field(default_factory=list)
    produces: list[str] = field(default_factory=list)

    def params(self) -> dict[str, Any]:
        return {}

    @abstractmethod
    def run(self, ctx: RunContext, exec_ctx: ExecContext) -> dict[str, ProducedArtifact]: ...


def compute_cache_key(process: ProcessBase, ctx: RunContext) -> str:
    required = []
    for k in process.requires:
        a = ctx.get(k)
        required.append((k, a.sha256, a.schema))

    optional = []
    for opt in process.optional:
        if not opt.affects_cache:
            continue

        a = ctx.get_optional(opt.key)
        if a:
            optional.append((opt.key, True, a.sha256, a.schema))
        else:
            optional.append((opt.key, False, None, None))

    payload = {
        "process": process.name,
        "version": process.version,
        "params": process.params(),
        "required": required,
        "optional": optional,
    }
    return sha256_bytes(stable_json_dumps(payload))


# ---------------------------------------------------------------------------
# Map / Reduce ディスクリプタ
# ---------------------------------------------------------------------------
_PROBE_SENTINEL = "__PROBE__"


@dataclass(frozen=True)
class Map:
    """variant ごとに process_class のインスタンスを生成する."""
    process_class: type[ProcessBase]
    key_prefix: str = ""
    kwargs: dict[str, Any] = field(default_factory=dict)
    kwargs_factory: Any = None  # Callable[[str], dict] | None

    def _resolve_kwargs(self, variant: str) -> dict[str, Any]:
        base = dict(self.kwargs)
        if self.kwargs_factory is not None:
            base.update(self.kwargs_factory(variant))
        return base

    def _infer_prefix(self) -> str:
        if self.key_prefix:
            return self.key_prefix
        probe = self.process_class(model_name=_PROBE_SENTINEL, **self._resolve_kwargs(_PROBE_SENTINEL))  # type: ignore[call-arg]
        for req in probe.requires:
            if req.endswith(f".{_PROBE_SENTINEL}"):
                return req.rsplit(".", 1)[0]
        raise ValueError(
            f"Cannot infer key_prefix for {self.process_class.__name__}. "
            f"Provide key_prefix explicitly."
        )

    def discover_variants(self, ctx: RunContext) -> list[str]:
        prefix = self._infer_prefix()
        variants = []
        for key in ctx.artifacts:
            if key.startswith(prefix + "."):
                variant = key[len(prefix) + 1:]
                if variant not in variants:
                    variants.append(variant)
        return variants

    def expand(self, ctx: RunContext) -> list[ProcessBase]:
        variants = self.discover_variants(ctx)
        return [self.process_class(model_name=v, **self._resolve_kwargs(v)) for v in variants]  # type: ignore[call-arg]


@dataclass(frozen=True)
class Reduce:
    """全 variant の成果物を集約する process_class のインスタンスを生成する."""
    process_class: type[ProcessBase]
    key_prefix: str = ""

    def _infer_prefix(self) -> str:
        if self.key_prefix:
            return self.key_prefix
        probe = self.process_class(model_names=[_PROBE_SENTINEL])  # type: ignore[call-arg]
        for req in probe.requires:
            if req.endswith(f".{_PROBE_SENTINEL}"):
                return req.rsplit(".", 1)[0]
        raise ValueError(
            f"Cannot infer key_prefix for {self.process_class.__name__}. "
            f"Provide key_prefix explicitly."
        )

    def discover_variants(self, ctx: RunContext) -> list[str]:
        prefix = self._infer_prefix()
        variants = []
        for key in ctx.artifacts:
            if key.startswith(prefix + "."):
                variant = key[len(prefix) + 1:]
                if variant not in variants:
                    variants.append(variant)
        return variants

    def expand(self, ctx: RunContext) -> ProcessBase:
        variants = self.discover_variants(ctx)
        return self.process_class(model_names=variants)  # type: ignore[call-arg]


class PipelineHalted(Exception):
    """Gate の条件未達によりパイプラインが途中停止したことを示す."""

    def __init__(self, message: str, ctx: RunContext) -> None:
        super().__init__(message)
        self.ctx = ctx


@dataclass(frozen=True)
class Gate:
    """条件を満たすまでパイプラインを停止するゲート.

    check が False を返すと PipelineHalted を送出する.
    再実行時にキャッシュ済みステップをスキップし、Gate を再評価する.
    """
    check: Any  # Callable[[RunContext], bool]  (dataclass + callable の型制約回避)
    message: str = "Pipeline halted: gate condition not met."


# ---------------------------------------------------------------------------
# Pipeline 内部の Phase 表現
# ---------------------------------------------------------------------------
Step = ProcessBase | Map | Reduce | Gate


@dataclass
class _StaticPhase:
    processes: list[ProcessBase]


@dataclass
class _ChainPhase:
    maps: list[Map]


@dataclass
class _ReducePhase:
    reduce: Reduce


@dataclass
class _GatePhase:
    gate: Gate


_Phase = _StaticPhase | _ChainPhase | _ReducePhase | _GatePhase


def _build_phases(steps: Sequence[Step]) -> list[_Phase]:
    phases: list[_Phase] = []
    for step in steps:
        if isinstance(step, ProcessBase):
            if phases and isinstance(phases[-1], _StaticPhase):
                phases[-1].processes.append(step)
            else:
                phases.append(_StaticPhase(processes=[step]))
        elif isinstance(step, Map):
            if phases and isinstance(phases[-1], _ChainPhase):
                phases[-1].maps.append(step)
            else:
                phases.append(_ChainPhase(maps=[step]))
        elif isinstance(step, Reduce):
            phases.append(_ReducePhase(reduce=step))
        elif isinstance(step, Gate):
            phases.append(_GatePhase(gate=step))
    return phases


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def _check_cache_static(proc: ProcessBase, ck: str, ctx: RunContext) -> bool:
    """静的 produces のキャッシュ判定."""
    for pk in proc.produces:
        art = ctx.artifacts.get(pk)
        if not art or art.cache_key != ck or not art.path.exists() or sha256_file(art.path) != art.sha256:
            return False
    return True


def _check_cache_dynamic(proc: ProcessBase, ck: str, ctx: RunContext) -> bool:
    """動的 produces (produces=[]) のキャッシュ判定. producer 名で検索."""
    prev = [a for a in ctx.artifacts.values() if a.producer == proc.name]
    if not prev:
        return False
    return all(
        a.cache_key == ck and a.path.exists() and sha256_file(a.path) == a.sha256
        for a in prev
    )


def _execute_one(
    proc: ProcessBase,
    ctx: RunContext,
    exec_ctx: ExecContext,
    *,
    force: set[str],
    relocate_to: Path | None = None,
) -> None:
    """単一プロセスの実行 (キャッシュ判定 → run → 検証 → 登録).

    relocate_to が指定された場合、生成ファイルをそのディレクトリに移動し
    Artifact のパスを移動先で登録する.
    """
    for req in proc.requires:
        ctx.get(req)

    ck = compute_cache_key(proc, ctx)

    if proc.produces:
        cache_hit = _check_cache_static(proc, ck, ctx)
    else:
        cache_hit = _check_cache_dynamic(proc, ck, ctx)

    if cache_hit and proc.name not in force:
        return

    produced = proc.run(ctx, exec_ctx)

    if proc.produces:
        produced_keys = set(produced.keys())
        expected_keys = set(proc.produces)
        if produced_keys != expected_keys:
            missing = expected_keys - produced_keys
            extra = produced_keys - expected_keys
            parts = [f"Process '{proc.name}' produces mismatch:"]
            if missing:
                parts.append(f"missing={missing}")
            if extra:
                parts.append(f"unexpected={extra}")
            raise RuntimeError(" ".join(parts))

    for key, part in produced.items():
        final_path = part.path
        if relocate_to is not None:
            final_path = relocate_to / part.path.name
            final_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(part.path), str(final_path))
        sha = sha256_file(final_path)
        ctx.put(
            Artifact(
                key=key,
                path=final_path,
                format=part.format,
                schema=part.schema,
                producer=proc.name,
                cache_key=ck,
                sha256=sha,
            )
        )


class Pipeline:
    def __init__(self, steps: Sequence[Step]) -> None:
        self._phases = _build_phases(steps)
        self._validate_steps(steps)

    def _validate_steps(self, steps: Sequence[Step]) -> None:
        names: list[str] = []
        for step in steps:
            if isinstance(step, ProcessBase):
                if not step.name:
                    raise ValueError("Process has empty name")
                names.append(step.name)
        if len(names) != len(set(names)):
            dupes = {n for n in names if names.count(n) > 1}
            raise ValueError(f"Duplicate process names: {dupes}")

    def run(
        self,
        ctx: RunContext,
        exec_ctx: ExecContext,
        *,
        force_processes: Sequence[str] | None = None,
    ) -> RunContext:
        force = set(force_processes or [])

        for phase in self._phases:
            if isinstance(phase, _StaticPhase):
                self._execute_static(phase.processes, ctx, exec_ctx, force)
            elif isinstance(phase, _ChainPhase):
                self._execute_chains(phase.maps, ctx, exec_ctx, force)
            elif isinstance(phase, _ReducePhase):
                proc = phase.reduce.expand(ctx)
                self._execute_static([proc], ctx, exec_ctx, force)
            elif isinstance(phase, _GatePhase):
                if not phase.gate.check(ctx):
                    raise PipelineHalted(phase.gate.message, ctx)

        return ctx

    def _execute_static(
        self,
        processes: list[ProcessBase],
        ctx: RunContext,
        exec_ctx: ExecContext,
        force: set[str],
    ) -> None:
        for proc in processes:
            _execute_one(proc, ctx, exec_ctx, force=force)

    def _execute_chains(
        self,
        maps: list[Map],
        ctx: RunContext,
        exec_ctx: ExecContext,
        force: set[str],
    ) -> None:
        variants = maps[0].discover_variants(ctx)
        if not variants:
            return

        chains: dict[str, list[ProcessBase]] = {}
        for variant in variants:
            chains[variant] = [m.process_class(model_name=variant, **m._resolve_kwargs(variant)) for m in maps]  # type: ignore[call-arg]

        def run_chain(variant: str, procs: list[ProcessBase]) -> None:
            chain_temp = exec_ctx.temp_dir / variant
            chain_temp.mkdir(parents=True, exist_ok=True)
            chain_exec_ctx = ExecContext(
                out_dir=chain_temp,
                temp_dir=chain_temp,
                logger=exec_ctx.logger,
                env=_SandboxedEnvironment(exec_ctx.env, cwd=chain_temp),
            )
            for proc in procs:
                try:
                    _execute_one(
                        proc, ctx, chain_exec_ctx,
                        force=force, relocate_to=exec_ctx.out_dir,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Process '{proc.name}' failed in chain '{variant}'"
                    ) from exc

        errors: dict[str, Exception] = {}
        with ThreadPoolExecutor() as pool:
            futures = {
                variant: pool.submit(run_chain, variant, procs)
                for variant, procs in chains.items()
            }
            for variant, future in futures.items():
                try:
                    future.result()
                except Exception as exc:
                    errors[variant] = exc
                    exec_ctx.logger.warning(
                        "Chain '%s' failed: %s", variant, exc,
                    )

        # 一時ディレクトリを削除 (成功・失敗問わず)
        for variant in variants:
            chain_temp = exec_ctx.temp_dir / variant
            if chain_temp.exists():
                shutil.rmtree(chain_temp)

        if errors:
            exec_ctx.logger.warning(
                "Partial failure: %d/%d variants failed: %s",
                len(errors), len(variants), list(errors.keys()),
            )
