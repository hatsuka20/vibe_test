"""recipe モジュールの単体テスト."""

from pathlib import Path

import json5
import pytest

from recipe import CompileOptions, ModelConfig, Recipe, RunOptions


class TestRecipeLoad:
    def test_load_minimal(self, tmp_path: Path) -> None:
        path = tmp_path / "recipe.json5"
        path.write_text('{ release: "v99" }')
        recipe = Recipe.load(path)
        assert recipe.release == "v99"
        assert recipe.compile_options.optimization_level == 2
        assert recipe.run_options.num_iterations == 100
        assert recipe.models == {}

    def test_load_full(self, tmp_path: Path) -> None:
        path = tmp_path / "recipe.json5"
        path.write_text(json5.dumps({
            "release": "v42",
            "compile_options": {"optimization_level": 3},
            "run_options": {"num_iterations": 200},
            "models": {
                "resnet": {
                    "compile_options": {"optimization_level": 1},
                },
                "vgg": {},
            },
        }))
        recipe = Recipe.load(path)
        assert recipe.release == "v42"
        assert recipe.compile_options.optimization_level == 3
        assert recipe.run_options.num_iterations == 200
        assert "resnet" in recipe.models
        assert "vgg" in recipe.models


class TestRecipeSave:
    def test_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "recipe.json5"
        original = Recipe(
            release="v10",
            compile_options=CompileOptions(optimization_level=3),
            models={"resnet": ModelConfig()},
        )
        original.save(path)

        loaded = Recipe.load(path)
        assert loaded.release == "v10"
        assert loaded.compile_options.optimization_level == 3
        assert "resnet" in loaded.models


class TestResolveOptions:
    def test_compile_default(self) -> None:
        """個別設定なし → 共通設定が使われる."""
        recipe = Recipe(compile_options=CompileOptions(optimization_level=3))
        opts = recipe.resolve_compile_options("resnet")
        assert opts.optimization_level == 3

    def test_compile_override(self) -> None:
        """個別設定あり → 共通設定をオーバーライド."""
        recipe = Recipe(
            compile_options=CompileOptions(optimization_level=2),
            models={
                "resnet": ModelConfig(
                    compile_options=CompileOptions(optimization_level=0),
                ),
            },
        )
        opts = recipe.resolve_compile_options("resnet")
        assert opts.optimization_level == 0

    def test_compile_unknown_model_uses_default(self) -> None:
        """models に存在しないモデル → 共通設定."""
        recipe = Recipe(compile_options=CompileOptions(optimization_level=3))
        opts = recipe.resolve_compile_options("unknown")
        assert opts.optimization_level == 3

    def test_run_default(self) -> None:
        recipe = Recipe(run_options=RunOptions(num_iterations=500))
        opts = recipe.resolve_run_options("vgg")
        assert opts.num_iterations == 500

    def test_run_override(self) -> None:
        recipe = Recipe(
            run_options=RunOptions(num_iterations=100),
            models={
                "vgg": ModelConfig(
                    run_options=RunOptions(num_iterations=1000),
                ),
            },
        )
        opts = recipe.resolve_run_options("vgg")
        assert opts.num_iterations == 1000


class TestPopulateModels:
    def test_adds_new_models(self) -> None:
        recipe = Recipe()
        changed = recipe.populate_models(["resnet", "vgg"])
        assert changed is True
        assert set(recipe.models.keys()) == {"resnet", "vgg"}

    def test_resets_confirmed_on_change(self) -> None:
        """新しいモデルが追加されたら confirmed が False にリセットされる."""
        recipe = Recipe(confirmed=True)
        recipe.populate_models(["resnet"])
        assert recipe.confirmed is False

    def test_no_change_if_already_present(self) -> None:
        recipe = Recipe(
            models={"resnet": ModelConfig(), "vgg": ModelConfig()},
            confirmed=True,
        )
        changed = recipe.populate_models(["resnet", "vgg"])
        assert changed is False
        assert recipe.confirmed is True  # 変更なし → confirmed 維持

    def test_preserves_existing_config(self) -> None:
        """既存の個別設定が上書きされないことを確認."""
        recipe = Recipe(
            models={
                "resnet": ModelConfig(
                    compile_options=CompileOptions(optimization_level=0),
                ),
            },
        )
        recipe.populate_models(["resnet", "vgg"])
        assert recipe.models["resnet"].compile_options.optimization_level == 0
        assert recipe.models["vgg"].compile_options is None


class TestModelsConfirmed:
    def test_not_confirmed_by_default(self) -> None:
        assert Recipe().models_confirmed() is False

    def test_confirmed_when_flag_set(self) -> None:
        recipe = Recipe(confirmed=True, models={"resnet": ModelConfig()})
        assert recipe.models_confirmed() is True

    def test_not_confirmed_even_with_models(self) -> None:
        """models があっても confirmed=False なら未確認."""
        recipe = Recipe(models={"resnet": ModelConfig()})
        assert recipe.models_confirmed() is False
