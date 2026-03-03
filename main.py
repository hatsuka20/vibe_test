import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence


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
    path: str
    format: str
    schema: str
    producer: str
    cache_key: str
    sha256: str
    meta: dict[str, Any] = field(default_factory=dict)

    def path_obj(self) -> Path:
        return Path(self.path)

    def exists(self) -> bool:
        return self.path_obj().exists()


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
        obj = {
            "run_dir": str(self.run_dir),
            "artifacts": {k: asdict(v) for k, v in self.artifacts.items()},
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
                arts[k] = Artifact(**v)
            ctx.artifacts = arts
        return ctx


@dataclass
class ExecContext:
    out_dir: Path
    temp_dir: Path
    logger: logging.Logger


class Process(Protocol):
    name: str
    requires: Sequence[str]
    optional: Sequence[OptionalInput]
    produces: Sequence[str]
    version: str

    def params(self) -> dict[str, Any]: ...

    def run(self, ctx: RunContext, exec_ctx: ExecContext) -> dict[str, tuple[Path, str, str]]: ...


def compute_cache_key(process: Process, ctx: RunContext) -> str:
    required = []
    for k in process.requires:
        a = ctx.get(k)
        required.append((k, a.sha256, a.schema))

    optional = []
    for opt in process.optional:
        a = ctx.get_optional(opt.key)

        if not opt.affects_cache:
            continue

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
    def __init__(self, processes: Sequence[Process]) -> None:
        self.processes = list(processes)

    def run(
        self,
        ctx: RunContext,
        exec_ctx: ExecContext,
        *,
        from_process: str | None = None,
        force_processes: Sequence[str] | None = None,
    ) -> RunContext:
        force = set(force_processes or [])
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
                if not art or art.cache_key != ck or not Path(art.path).exists():
                    cache_hit = False
                    break

            if cache_hit and proc.name not in force:
                continue

            produced = proc.run(ctx, exec_ctx)

            for key, (path, fmt, schema) in produced.items():
                sha = sha256_file(path)
                ctx.put(
                    Artifact(
                        key=key,
                        path=str(path),
                        format=fmt,
                        schema=schema,
                        producer=proc.name,
                        cache_key=ck,
                        sha256=sha,
                        meta={"ts": time.time()},
                    )
                )

        return ctx

