"""レシピファイルの読み込みと per-model 設定の解決."""

from pathlib import Path

import json5
from pydantic import BaseModel, Field


class CompileOptions(BaseModel):
    optimization_level: int = 2


class RunOptions(BaseModel):
    num_iterations: int = 100


class ModelConfig(BaseModel):
    """モデル個別設定. 未指定のフィールドは共通設定で埋められる."""
    compile_options: CompileOptions | None = None
    run_options: RunOptions | None = None


class Recipe(BaseModel):
    release: str = "v50"
    compile_options: CompileOptions = Field(default_factory=CompileOptions)
    run_options: RunOptions = Field(default_factory=RunOptions)
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    confirmed: bool = False

    @classmethod
    def load(cls, path: Path) -> "Recipe":
        data = json5.loads(path.read_text(encoding="utf-8"))
        return cls.model_validate(data)

    def save(self, path: Path) -> None:
        path.write_text(
            json5.dumps(self.model_dump(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def resolve_compile_options(self, model_name: str) -> CompileOptions:
        """共通設定に個別設定をオーバーライドして返す."""
        base = self.compile_options.model_dump()
        model_cfg = self.models.get(model_name)
        if model_cfg and model_cfg.compile_options:
            base.update(
                {k: v for k, v in model_cfg.compile_options.model_dump().items()
                 if v is not None}
            )
        return CompileOptions.model_validate(base)

    def resolve_run_options(self, model_name: str) -> RunOptions:
        """共通設定に個別設定をオーバーライドして返す."""
        base = self.run_options.model_dump()
        model_cfg = self.models.get(model_name)
        if model_cfg and model_cfg.run_options:
            base.update(
                {k: v for k, v in model_cfg.run_options.model_dump().items()
                 if v is not None}
            )
        return RunOptions.model_validate(base)

    def populate_models(self, model_names: list[str]) -> bool:
        """ダウンロードで発見されたモデル名をレシピに反映する.

        新しいモデルが追加された場合 confirmed を False にリセットし True を返す.
        """
        changed = False
        for name in model_names:
            if name not in self.models:
                self.models[name] = ModelConfig()
                changed = True
        if changed:
            self.confirmed = False
        return changed

    def models_confirmed(self) -> bool:
        """モデルリストが確認済みかを返す."""
        return self.confirmed
