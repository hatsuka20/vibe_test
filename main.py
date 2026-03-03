"""モックパイプラインのエントリポイント."""

import argparse
import logging
from pathlib import Path

from environment import DryRunEnvironment
from pipeline import ExecContext, Pipeline, RunContext
from mock_pipeline import CompileModel, DownloadModel, FormatProfile, RunModel

class Args(argparse.Namespace):
    experiment_name: str

def main() -> None:
    parser = argparse.ArgumentParser(description="モックパイプラインの実行")
    parser.add_argument("experiment_name", help="実験名 (experiments/<name> に出力)")
    args = parser.parse_args(namespace=Args())

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("mock_pipeline")

    base_dir = Path("experiments") / args.experiment_name
    run_dir = base_dir / "run"
    out_dir = base_dir / "out"
    temp_dir = base_dir / "tmp"
    for d in (out_dir, temp_dir):
        d.mkdir(parents=True, exist_ok=True)

    env = DryRunEnvironment()
    ctx = RunContext.load(run_dir=run_dir)
    exec_ctx = ExecContext(out_dir=out_dir, temp_dir=temp_dir, logger=logger, env=env)

    pipeline = Pipeline([
        DownloadModel(),
        CompileModel(),
        RunModel(),
        FormatProfile(),
    ])

    ctx = pipeline.run(ctx, exec_ctx)

    logger.info("Manifest: %s", ctx.manifest_path)
    for cmd in env.history:
        logger.info("Command: %s", " ".join(cmd.build()))


if __name__ == "__main__":
    main()
