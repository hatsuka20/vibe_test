"""ターゲットチップに応じた内部パラメータの解決.

ユーザはレシピで chip 名 (chipX, chipY, ...) のみ指定する.
Toolchain がチップ名から具体的なリンクライブラリ・コンパイラフラグ等を解決する.
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


class Toolchain:
    """チップ名からコンパイル・ランタイムの内部パラメータを解決する."""

    def __init__(self, chip: str) -> None:
        if chip not in _CHIP_PROFILES:
            raise ValueError(
                f"Unknown chip: {chip!r}. "
                f"Available: {', '.join(sorted(_CHIP_PROFILES))}"
            )
        self._profile = _CHIP_PROFILES[chip]

    @property
    def chip(self) -> str:
        return self._profile.chip

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
