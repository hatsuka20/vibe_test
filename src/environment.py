"""コマンド実行環境の抽象化.

- Environment: コマンドを「どこで」実行するか (Local / Remote / DryRun)
- CommandBuilder: コマンドを「どう組み立てるか」
"""

from __future__ import annotations

import shlex
import subprocess
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    returncode: int
    stdout: bytes
    stderr: bytes

    def check(self) -> None:
        """returncode != 0 なら CalledProcessError を送出する."""
        if self.returncode != 0:
            raise subprocess.CalledProcessError(
                self.returncode, self.command, self.stdout, self.stderr,
            )


class CommandBuilder(ABC):
    @abstractmethod
    def build(self) -> list[str]: ...


class Environment(ABC):
    @abstractmethod
    def run(self, command: CommandBuilder, *, cwd: Path | None = None) -> CommandResult: ...


class LocalEnvironment(Environment):
    def __init__(self, timeout: float = 1000) -> None:
        self.timeout = timeout

    def run(self, command: CommandBuilder, *, cwd: Path | None = None) -> CommandResult:
        argv = command.build()
        result = subprocess.run(argv, cwd=cwd, capture_output=True, timeout=self.timeout)
        return CommandResult(
            command=argv,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )


class RemoteEnvironment(Environment):
    def __init__(self, host: str, user: str | None = None, timeout: float = 1000) -> None:
        self.host = host
        self.user = user
        self.timeout = timeout

    @property
    def _target(self) -> str:
        return f"{self.user}@{self.host}" if self.user else self.host

    def run(self, command: CommandBuilder, *, cwd: Path | None = None) -> CommandResult:
        argv = command.build()
        remote_cmd = " ".join(shlex.quote(c) for c in argv)
        if cwd:
            remote_cmd = f"cd {shlex.quote(str(cwd))} && {remote_cmd}"
        ssh_command = ["ssh", self._target, remote_cmd]
        result = subprocess.run(ssh_command, capture_output=True, timeout=self.timeout)
        return CommandResult(
            command=argv,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )


@dataclass(frozen=True)
class DryRunRecord:
    """DryRunEnvironment が記録する 1 回分の呼び出し."""
    command: CommandBuilder
    cwd: Path | None


class DryRunEnvironment(Environment):
    """コマンドを記録するだけで実行しない. テストや検証用."""

    def __init__(self) -> None:
        self.history: list[CommandBuilder] = []
        self.records: list[DryRunRecord] = []
        self._lock = threading.Lock()

    def run(self, command: CommandBuilder, *, cwd: Path | None = None) -> CommandResult:
        with self._lock:
            self.history.append(command)
            self.records.append(DryRunRecord(command=command, cwd=cwd))
        return CommandResult(
            command=command.build(),
            returncode=0,
            stdout=b"",
            stderr=b"",
        )


