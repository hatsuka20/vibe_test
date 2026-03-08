"""toolchain モジュールの単体テスト."""

from pathlib import Path

import pytest

from toolchain import ChipProfile, MachineSpec, Toolchain


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


class TestToolsetVersion:
    def test_old_version_uses_hoge_tools(self) -> None:
        """2.43.0 より前は /opt/hoge_tools."""
        tc = Toolchain("chipX", toolset_version="2.40.0")
        assert tc.tools_dir == Path("/opt/hoge_tools")
        assert tc.compiler_path == Path("/opt/hoge_tools/bin/model-compiler")
        assert tc.runtime_path == Path("/opt/hoge_tools/bin/model-runtime")

    def test_new_version_uses_fuga_tools(self) -> None:
        """2.43.0 以降は /opt/fuga_tools."""
        tc = Toolchain("chipX", toolset_version="2.43.0")
        assert tc.tools_dir == Path("/opt/fuga_tools")
        assert tc.compiler_path == Path("/opt/fuga_tools/bin/model-compiler")
        assert tc.runtime_path == Path("/opt/fuga_tools/bin/model-runtime")

    def test_newer_version_uses_fuga_tools(self) -> None:
        tc = Toolchain("chipY", toolset_version="3.0.0")
        assert tc.tools_dir == Path("/opt/fuga_tools")

    def test_version_preserved(self) -> None:
        tc = Toolchain("chipX", toolset_version="2.41.5")
        assert tc.toolset_version == "2.41.5"


class TestMachineMapping:
    def test_chipX_runs_on_m1(self) -> None:
        tc = Toolchain("chipX")
        assert tc.machine.host == "m1.example.com"

    def test_chipY_runs_on_m1(self) -> None:
        tc = Toolchain("chipY")
        assert tc.machine.host == "m1.example.com"

    def test_chipZ_runs_on_m2(self) -> None:
        tc = Toolchain("chipZ")
        assert tc.machine.host == "m2.example.com"

    def test_machine_has_user(self) -> None:
        tc = Toolchain("chipX")
        assert tc.machine.user == "root"

    def test_default_port(self) -> None:
        tc = Toolchain("chipX")
        assert tc.machine.port == 22102

    def test_custom_port(self) -> None:
        tc = Toolchain("chipX", port=22108)
        assert tc.machine.port == 22108
        assert tc.machine.host == "m1.example.com"
