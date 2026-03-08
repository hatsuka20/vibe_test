"""モックパイプラインのエントリポイント."""

import argparse
import logging
import shutil
from pathlib import Path

from environment import DryRunEnvironment
from pipeline import ExecContext, Gate, Map, Pipeline, PipelineHalted, Reduce, RunContext
from processes import (
    AggregateProfile,
    CompileModel,
    DownloadModel,
    FormatProfile,
    RunModel,
)
from recipe import Recipe


class Args(argparse.Namespace):
    experiment_name: str
    recipe: str | None


def _resolve_recipe(
    template_path: Path | None, base_dir: Path, logger: logging.Logger,
) -> tuple[Recipe, Path]:
    """experiment ディレクトリ内のレシピを解決する.

    - experiment 内にレシピがあればそれを使う (--recipe 不要)
    - なければ --recipe で指定されたテンプレートからコピーして使う
    - --recipe が experiment 内のレシピ自身を指していればそのまま使う
    - 既にレシピがある状態で --recipe を指定したらエラー
    """
    experiment_recipe_path = base_dir / "recipe.json5"

    if experiment_recipe_path.exists():
        if template_path is not None:
            # --recipe が experiment 内のレシピ自身を指している場合は許容
            if template_path.resolve() != experiment_recipe_path.resolve():
                raise SystemExit(
                    f"エラー: 既にレシピが存在します: {experiment_recipe_path}\n"
                    "--recipe を指定せずに再実行してください。"
                )
        logger.info("既存レシピを使用: %s", experiment_recipe_path)
        return Recipe.load(experiment_recipe_path), experiment_recipe_path

    # experiment 内にレシピがない → --recipe 必須
    if template_path is None:
        raise SystemExit(
            f"エラー: レシピが見つかりません: {experiment_recipe_path}\n"
            "--recipe でテンプレートレシピのパスを指定してください。"
        )

    # テンプレートからコピー
    base_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(template_path), str(experiment_recipe_path))
    logger.info("テンプレートからレシピをコピー: %s -> %s", template_path, experiment_recipe_path)
    return Recipe.load(experiment_recipe_path), experiment_recipe_path


def main() -> None:
    parser = argparse.ArgumentParser(description="モックパイプラインの実行")
    parser.add_argument("experiment_name", help="実験名 (experiments/<name> に出力)")
    parser.add_argument("--recipe", default=None, help="テンプレートレシピのパス (初回のみ必須)")
    args = parser.parse_args(namespace=Args())

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("mock_pipeline")

    base_dir = Path("experiments") / args.experiment_name
    run_dir = base_dir / "run"
    out_dir = base_dir / "out"
    temp_dir = base_dir / "tmp"
    for d in (out_dir, temp_dir):
        d.mkdir(parents=True, exist_ok=True)

    template_path = Path(args.recipe) if args.recipe else None
    recipe, recipe_path = _resolve_recipe(template_path, base_dir, logger)

    env = DryRunEnvironment()
    ctx = RunContext.load(run_dir=run_dir)
    exec_ctx = ExecContext(out_dir=out_dir, temp_dir=temp_dir, logger=logger, env=env)

    pipeline = Pipeline([
        DownloadModel(
            release=recipe.release,
            recipe=recipe,
            recipe_path=recipe_path,
        ),
        Gate(
            check=lambda ctx: recipe.models_confirmed(),
            message=(
                f"レシピにモデル名が反映されました: {recipe_path}\n"
                "各モデルの設定を確認・編集してから再実行してください。"
            ),
        ),
        Map(CompileModel, kwargs={"recipe": recipe}),
        Map(RunModel, kwargs={"recipe": recipe}),
        Map(FormatProfile),
        Reduce(AggregateProfile),
    ])

    try:
        ctx = pipeline.run(ctx, exec_ctx)
    except PipelineHalted as e:
        logger.warning("%s", e)
        return


if __name__ == "__main__":
    main()
