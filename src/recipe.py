"""レシピファイルの読み込みと per-model 設定の解決."""

from pathlib import Path

import json5
from pydantic import BaseModel, Field


class CompileOptions(BaseModel):
    optimization_level: int = 2


class RunOptions(BaseModel):
    num_iterations: int = 100


class TargetConfig(BaseModel):
    """ターゲットチップの指定. ユーザはチップ名のみ指定する."""
    chip: str = "chipX"


class ModelConfig(BaseModel):
    """モデル個別設定. 未指定のフィールドは共通設定で埋められる."""
    name: str
    compile_options: CompileOptions | None = None
    run_options: RunOptions | None = None


class Recipe(BaseModel):
    release: str = "v50"
    url_base: str = "https://example.com/models"
    target: TargetConfig = Field(default_factory=TargetConfig)
    compile_options: CompileOptions = Field(default_factory=CompileOptions)
    run_options: RunOptions = Field(default_factory=RunOptions)
    models: list[ModelConfig] = Field(default_factory=list)
    confirmed: bool = True

    @classmethod
    def load(cls, path: Path) -> "Recipe":
        data = json5.loads(path.read_text(encoding="utf-8"))
        return cls.model_validate(data)

    def save(self, path: Path) -> None:
        path.write_text(
            json5.dumps(self.model_dump(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def get_model(self, model_name: str) -> ModelConfig | None:
        """名前でモデル設定を検索する."""
        for m in self.models:
            if m.name == model_name:
                return m
        return None

    def model_names(self) -> list[str]:
        """全モデル名をリストで返す."""
        return [m.name for m in self.models]

    def resolve_compile_options(self, model_name: str) -> CompileOptions:
        """共通設定に個別設定をオーバーライドして返す."""
        base = self.compile_options.model_dump()
        model_cfg = self.get_model(model_name)
        if model_cfg and model_cfg.compile_options:
            base.update(
                {k: v for k, v in model_cfg.compile_options.model_dump().items()
                 if v is not None}
            )
        return CompileOptions.model_validate(base)

    def resolve_run_options(self, model_name: str) -> RunOptions:
        """共通設定に個別設定をオーバーライドして返す."""
        base = self.run_options.model_dump()
        model_cfg = self.get_model(model_name)
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
        existing = {m.name for m in self.models}
        changed = False
        for name in model_names:
            if name not in existing:
                self.models.append(ModelConfig(name=name))
                changed = True
        if changed:
            self.confirmed = False
        return changed

    def models_confirmed(self) -> bool:
        """モデルリストが確認済みかを返す."""
        return self.confirmed
