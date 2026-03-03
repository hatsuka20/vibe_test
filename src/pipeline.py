import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from environment import Environment, LocalEnvironment


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

    def __post_init__(self) -> None:
        self.manifest_path = self.run_dir / "manifest.json"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Artifact:
        if key not in self.artifacts:
            raise KeyError(f"Missing artifact: {key}")
        return self.artifacts[key]

    def get_optional(self, key: str) -> Artifact | None:
        return self.artifacts.get(key)

    def put(self, art: Artifact) -> None:
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
    """Process の基底クラス. 共通フィールドのデフォルトを提供する."""

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


class Pipeline:
    def __init__(self, processes: Sequence[ProcessBase]) -> None:
        self.processes = list(processes)
        names = [p.name for p in self.processes]
        empty = [i for i, n in enumerate(names) if not n]
        if empty:
            raise ValueError(f"Process at index {empty} has empty name")
        if len(names) != len(set(names)):
            dupes = {n for n in names if names.count(n) > 1}
            raise ValueError(f"Duplicate process names: {dupes}")

    def run(
        self,
        ctx: RunContext,
        exec_ctx: ExecContext,
        *,
        from_process: str | None = None,
        force_processes: Sequence[str] | None = None,
    ) -> RunContext:
        force = set(force_processes or [])
        all_names = {p.name for p in self.processes}
        unknown = force - all_names
        if unknown:
            raise ValueError(f"Unknown force_processes: {unknown}")

        start_idx = 0
        if from_process is not None:
            names = [p.name for p in self.processes]
            if from_process not in names:
                raise ValueError(f"Unknown from_process: {from_process}")
            start_idx = names.index(from_process)

        for i, proc in enumerate(self.processes):
            if i < start_idx:
                continue

            for req in proc.requires:
                ctx.get(req)

            ck = compute_cache_key(proc, ctx)

            cache_hit = True
            for pk in proc.produces:
                art = ctx.artifacts.get(pk)
                if not art or art.cache_key != ck or not art.path.exists() or sha256_file(art.path) != art.sha256:
                    cache_hit = False
                    break

            if cache_hit and proc.name not in force:
                continue

            produced = proc.run(ctx, exec_ctx)

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
                sha = sha256_file(part.path)
                ctx.put(
                    Artifact(
                        key=key,
                        path=part.path,
                        format=part.format,
                        schema=part.schema,
                        producer=proc.name,
                        cache_key=ck,
                        sha256=sha,
                    )
                )

        return ctx
