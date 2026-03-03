"""コマンド実行環境の抽象化.

- Environment: コマンドを「どこで」実行するか (Local / Remote / DryRun)
- CommandBuilder: コマンドを「どう組み立てるか」
"""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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


class Environment(ABC):
    @abstractmethod
    def run(self, command: list[str], *, cwd: Path | None = None) -> CommandResult: ...


class LocalEnvironment(Environment):
    def run(self, command: list[str], *, cwd: Path | None = None) -> CommandResult:
        result = subprocess.run(command, cwd=cwd, capture_output=True)
        return CommandResult(
            command=command,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )


class RemoteEnvironment(Environment):
    def __init__(self, host: str, user: str | None = None) -> None:
        self.host = host
        self.user = user

    @property
    def _target(self) -> str:
        return f"{self.user}@{self.host}" if self.user else self.host

    def run(self, command: list[str], *, cwd: Path | None = None) -> CommandResult:
        remote_cmd = " ".join(command)
        if cwd:
            remote_cmd = f"cd {cwd} && {remote_cmd}"
        ssh_command = ["ssh", self._target, remote_cmd]
        result = subprocess.run(ssh_command, capture_output=True)
        return CommandResult(
            command=command,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )


class DryRunEnvironment(Environment):
    """コマンドを記録するだけで実行しない. テストや検証用."""

    def __init__(self) -> None:
        self.history: list[list[str]] = []

    def run(self, command: list[str], *, cwd: Path | None = None) -> CommandResult:
        self.history.append(command)
        return CommandResult(
            command=command,
            returncode=0,
            stdout=b"",
            stderr=b"",
        )


class CommandBuilder(ABC):
    @abstractmethod
    def build(self) -> list[str]: ...
