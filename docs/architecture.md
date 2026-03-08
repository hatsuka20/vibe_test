# Pipeline Artifact Model アーキテクチャ

## 概要

モデルのダウンロード・コンパイル・実行・プロファイリングを自動化するパイプラインフレームワーク。
アーティファクトベースのキャッシュ機構と、Map/Reduce による並列処理をサポートする。

## パイプライン全体像

```mermaid
flowchart TD
    DL["DownloadModel<br/><i>動的 produces</i>"]
    G["Gate<br/><i>confirmed チェック</i>"]
    GC["Map(GenerateConfig)"]
    CM["Map(CompileModel)"]
    RM["Map(RunModel)"]
    FP["Map(FormatProfile)"]
    AG["Reduce(AggregateProfile)"]

    DL --> G --> GC --> CM --> RM --> FP --> AG

    style DL fill:#e8f4fd,stroke:#2196F3
    style G fill:#fff3e0,stroke:#FF9800
    style GC fill:#e8f5e9,stroke:#4CAF50
    style CM fill:#e8f5e9,stroke:#4CAF50
    style RM fill:#e8f5e9,stroke:#4CAF50
    style FP fill:#e8f5e9,stroke:#4CAF50
    style AG fill:#f3e5f5,stroke:#9C27B0
```

### 実行時の動作

DownloadModel がモデルを動的に発見した後、連続する Map は **per-variant チェーン** に融合され並列実行される。

```mermaid
flowchart LR
    subgraph "逐次実行"
        DL[DownloadModel]
        G[Gate]
    end

    subgraph "並列実行 (ThreadPoolExecutor)"
        direction TB
        subgraph "Thread 1: resnet"
            GC1[GenerateConfig] --> CM1[CompileModel] --> RM1[RunModel] --> FP1[FormatProfile]
        end
        subgraph "Thread 2: vgg"
            GC2[GenerateConfig] --> CM2[CompileModel] --> RM2[RunModel] --> FP2[FormatProfile]
        end
    end

    subgraph "逐次実行 "
        AG[AggregateProfile]
    end

    DL --> G --> GC1 & GC2
    FP1 & FP2 --> AG
```

## レイヤー構成

ユーザーが触る設定から実行環境まで、4層に分離されている。

```mermaid
flowchart TB
    subgraph "Layer 1: Recipe (ユーザー設定)"
        R["recipe.json5<br/>chip, toolset_version, port<br/>compile_options, run_options<br/>models[]"]
    end

    subgraph "Layer 2: Toolchain (自動解決)"
        T["Toolchain<br/>chip → compiler_path, compile_lib,<br/>compile_flags, runtime_path, ...<br/>port → MachineSpec"]
    end

    subgraph "Layer 3: Process (実行ロジック)"
        P["GenerateConfig / CompileModel / RunModel / ...<br/>アーティファクトの生成・変換"]
    end

    subgraph "Layer 4: Environment (実行場所)"
        E["LocalEnvironment / RemoteEnvironment / DryRunEnvironment<br/>CommandBuilder → subprocess / ssh / 記録のみ"]
    end

    R -- "resolve_compile_options(name)" --> T
    T -- "kwargs_factory 経由" --> P
    P -- "env.run(cmd)" --> E
```

### 各レイヤーの責務

| レイヤー | 知っていること | 知らないこと |
|---------|-------------|------------|
| **Recipe** | ユーザーの意図 (チップ名, 最適化レベル, イテレーション数) | コンパイラパス, ライブラリ名 |
| **Toolchain** | チップ名 → 内部パラメータの対応 | どのモデルをコンパイルするか |
| **Process** | 入力アーティファクト → 出力アーティファクトの変換 | チップの種類 (config は artifact 経由) |
| **Environment** | コマンドをどこで実行するか | コマンドの意味 |

## アーティファクトの流れ

```mermaid
flowchart LR
    subgraph "per-model アーティファクト"
        M["model.resnet<br/><i>.onnx</i>"]
        C["config.resnet<br/><i>.ini / .json</i>"]
        CM["compiled_model.resnet<br/><i>.cpp</i>"]
        P["profile.resnet<br/><i>.json</i>"]
        R["report.resnet<br/><i>.txt</i>"]
    end
    S["summary_report<br/><i>.txt</i>"]

    M --> C & CM
    C --> CM
    CM --> P --> R --> S
```

各アーティファクトは `Artifact` として RunContext に登録され、キャッシュキー (SHA-256) で同一性が管理される。
プロセスの入力が変わらなければ再実行をスキップする。

## チップ固有の config 生成

GenerateConfig は CompileOptions とチップ名から、チップ固有フォーマットの config ファイルを生成する。
CompileModel はフォーマットを知らず、`--config <path>` として受け取るだけ。

```mermaid
flowchart LR
    CO["CompileOptions<br/>memory_mode, quantization_bits, ..."]

    subgraph "GenerateConfig"
        direction TB
        CX["chipX → INI"]
        CY["chipY → JSON"]
        CZ["chipZ → JSON"]
    end

    CF["config.{model}<br/><i>artifact</i>"]
    CModel["CompileModel<br/><code>--config path</code>"]

    CO --> CX & CY & CZ --> CF --> CModel
    CO -- "optimization_level" --> CModel
```

## フレームワークのコア概念

### ProcessBase

すべてのプロセスの基底クラス。

```
ProcessBase
├── name: str              プロセス名 (キャッシュキーの一部)
├── version: str           実装バージョン (bump → キャッシュ無効化)
├── requires: list[str]    入力アーティファクトのキー
├── produces: list[str]    出力アーティファクトのキー (空 = 動的)
├── params() → dict        キャッシュキーに含まれるパラメータ
└── run(ctx, exec_ctx) → dict[str, ProducedArtifact]
```

### Map / Reduce

```
Map(ProcessClass, kwargs_factory=...)
  → variant ごとに ProcessClass(model_name=variant, **kwargs) を生成
  → 連続する Map はチェーンに融合され並列実行

Reduce(ProcessClass)
  → 全 variant を集約する ProcessClass(model_names=[...]) を生成
```

### Phase 分割

Pipeline は steps を以下の Phase に分割して実行する:

| Phase | 構成要素 | 実行方式 |
|-------|---------|---------|
| StaticPhase | 連続する ProcessBase | 逐次実行 |
| ChainPhase | 連続する Map | per-variant 並列実行 |
| ReducePhase | Reduce | 逐次実行 |
| GatePhase | Gate | 条件チェック (未達で停止) |

### Gate

`check(ctx)` が `False` を返すと `PipelineHalted` を送出してパイプラインを停止する。
DownloadModel がモデル名をレシピに書き戻した後、ユーザーに設定確認を促すために使用される。

## ディレクトリ構成

```
pipeline_artifact_model/
├── src/
│   ├── pipeline.py       フレームワーク本体 (Artifact, RunContext, Map/Reduce, Pipeline)
│   ├── environment.py    コマンド実行環境 (Local / Remote / DryRun)
│   ├── recipe.py         レシピモデル (CompileOptions, RunOptions, Recipe)
│   ├── toolchain.py      チップ → 内部パラメータ解決 (Toolchain, ChipProfile)
│   ├── processes.py      パイプライン構成プロセス群
│   └── main.py           エントリポイント
├── recipes/
│   └── recipe.json5      テンプレートレシピ
├── tests/
│   ├── test_pipeline.py
│   ├── test_processes.py
│   ├── test_recipe.py
│   └── test_toolchain.py
└── experiments/           実行時に生成される実験ディレクトリ
    └── <name>/
        ├── recipe.json5   コピーされたレシピ
        ├── run/           RunContext (manifest.json)
        ├── out/           出力アーティファクト
        └── tmp/           一時ファイル (チェーンごとに分離)
```
