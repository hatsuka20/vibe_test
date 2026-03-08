"""ターゲットチップとツールセットバージョンに応じた内部パラメータの解決.

ユーザはレシピで chip 名 (chipX, chipY, ...) のみ指定する.
Toolchain がチップ名 + ツールセットバージョンから具体的なパス・
リンクライブラリ・コンパイラフラグ等を解決する.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ChipProfile:
    """チップ固有の内部パラメータ."""
    chip: str
    compile_lib: str         # コンパイル時にリンクするライブラリ (e.g. libChipX.so)
    compile_flags: list[str] # チップ固有のコンパイラフラグ
    runtime_lib: str         # 実行時にリンクするライブラリ
    runtime_flags: list[str] # チップ固有のランタイムフラグ


# チップ名 → 内部パラメータの静的マッピング
_CHIP_PROFILES: dict[str, ChipProfile] = {
    "chipX": ChipProfile(
        chip="chipX",
        compile_lib="libChipX.so",
        compile_flags=["--target=chipx"],
        runtime_lib="libChipXRuntime.so",
        runtime_flags=["--device=chipx"],
    ),
    "chipY": ChipProfile(
        chip="chipY",
        compile_lib="libChipY.so",
        compile_flags=["--target=chipy", "--fp16"],
        runtime_lib="libChipYRuntime.so",
        runtime_flags=["--device=chipy"],
    ),
    "chipZ": ChipProfile(
        chip="chipZ",
        compile_lib="libChipZ.so",
        compile_flags=["--target=chipz", "--int8"],
        runtime_lib="libChipZRuntime.so",
        runtime_flags=["--device=chipz"],
    ),
}


def _resolve_tools_dir(toolset_version: str) -> Path:
    """ツールセットバージョンからインストールディレクトリを解決する."""
    major, minor, _patch = (int(x) for x in toolset_version.split("."))
    if (major, minor) >= (2, 43):
        return Path("/opt/fuga_tools")
    return Path("/opt/hoge_tools")


class Toolchain:
    """チップ名 + ツールセットバージョンからパラメータを解決する."""

    def __init__(self, chip: str, toolset_version: str = "2.40.0") -> None:
        if chip not in _CHIP_PROFILES:
            raise ValueError(
                f"Unknown chip: {chip!r}. "
                f"Available: {', '.join(sorted(_CHIP_PROFILES))}"
            )
        self._profile = _CHIP_PROFILES[chip]
        self._toolset_version = toolset_version
        self._tools_dir = _resolve_tools_dir(toolset_version)

    @property
    def chip(self) -> str:
        return self._profile.chip

    @property
    def toolset_version(self) -> str:
        return self._toolset_version

    @property
    def tools_dir(self) -> Path:
        """ツールセットのインストールディレクトリ."""
        return self._tools_dir

    @property
    def compiler_path(self) -> Path:
        """コンパイラの実行パス."""
        return self._tools_dir / "bin" / "model-compiler"

    @property
    def runtime_path(self) -> Path:
        """ランタイムの実行パス."""
        return self._tools_dir / "bin" / "model-runtime"

    @property
    def compile_lib(self) -> str:
        return self._profile.compile_lib

    @property
    def compile_flags(self) -> list[str]:
        return self._profile.compile_flags

    @property
    def runtime_lib(self) -> str:
        return self._profile.runtime_lib

    @property
    def runtime_flags(self) -> list[str]:
        return self._profile.runtime_flags
