"""toolchain モジュールの単体テスト."""

import pytest

from toolchain import ChipProfile, Toolchain


class TestToolchain:
    def test_chipX(self) -> None:
        tc = Toolchain("chipX")
        assert tc.chip == "chipX"
        assert tc.compile_lib == "libChipX.so"
        assert tc.runtime_lib == "libChipXRuntime.so"

    def test_chipY(self) -> None:
        tc = Toolchain("chipY")
        assert tc.chip == "chipY"
        assert tc.compile_lib == "libChipY.so"
        assert "--fp16" in tc.compile_flags

    def test_chipZ(self) -> None:
        tc = Toolchain("chipZ")
        assert tc.chip == "chipZ"
        assert "--int8" in tc.compile_flags

    def test_unknown_chip_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown chip"):
            Toolchain("chipW")

    def test_compile_flags_are_chip_specific(self) -> None:
        """異なるチップで異なるフラグが返る."""
        tc_x = Toolchain("chipX")
        tc_y = Toolchain("chipY")
        assert tc_x.compile_flags != tc_y.compile_flags

    def test_runtime_flags_are_chip_specific(self) -> None:
        tc_x = Toolchain("chipX")
        tc_y = Toolchain("chipY")
        assert tc_x.runtime_flags != tc_y.runtime_flags
