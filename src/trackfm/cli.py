"""TrackFM command-line interface.

Subcommands mirror the project stages:
  download     Fetch raw DMA AIS zips
  clean        Run the cleaning pipeline (raw zips -> partitioned parquet)
  extract-aux  Extract vessel-intrinsic aux-field change-logs from raw zips
  materialize  Window + shuffle cleaned tracks into training shards
  pretrain     Pretrain the foundation model
  finetune     Fine-tune on a downstream task
  evaluate     Evaluate a checkpoint
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def download(
    start_date: str = typer.Option(..., help="YYYY-MM-DD"),
    end_date: str = typer.Option(..., help="YYYY-MM-DD"),
    raw_dir: Path = typer.Option(Path("~/data/ais/raw"), help="Destination directory"),
    force: bool = typer.Option(False, help="Re-download existing files"),
):
    """Download raw AIS zips from the Danish Maritime Authority."""
    from trackfm.data.download import AISDataDownloader, parse_date

    downloader = AISDataDownloader(raw_dir.expanduser())
    downloader.download_date_range(parse_date(start_date), parse_date(end_date), force)


@app.command()
def clean(
    config: Path = typer.Option(Path("configs/data/clean_dk.yaml"), help="Cleaning config YAML"),
    resume: bool = typer.Option(True, help="Resume from checkpoint"),
    max_files: Optional[int] = typer.Option(None, help="Limit number of zips (smoke test)"),
    reset: bool = typer.Option(False, help="Reset checkpoint/state before running"),
):
    """Run the AIS cleaning pipeline over raw zips."""
    from trackfm.data.config import load_config
    from trackfm.data.pipeline import AISPipeline

    cfg = load_config(str(config))
    pipeline = AISPipeline(cfg)
    stats = pipeline.run(resume=resume and not reset, max_files=max_files)
    typer.echo(stats)


@app.command("extract-aux")
def extract_aux(
    raw_dir: Path = typer.Option(Path("~/data/ais/raw"), help="Raw DMA zip directory"),
    aux_dir: Path = typer.Option(Path("~/data/ais/aux"), help="Output directory"),
    state_dir: Path = typer.Option(Path("~/data/ais/state/aux"), help="Checkpoint directory"),
    resume: bool = typer.Option(True, help="Skip days/zips already processed"),
    max_files: Optional[int] = typer.Option(None, help="Limit number of zips (smoke test)"),
    day: Optional[list[str]] = typer.Option(
        None, help="Restrict to YYYY-MM-DD day(s); repeatable (smoke test)"
    ),
):
    """Extract vessel-intrinsic aux-field change-logs from raw zips."""
    import logging

    from trackfm.data.aux_fields import extract_aux_fields

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    stats = extract_aux_fields(
        raw_dir.expanduser(),
        aux_dir.expanduser(),
        state_dir.expanduser(),
        resume=resume,
        max_files=max_files,
        days=day,
    )
    typer.echo(stats)


@app.command()
def materialize(
    config: Path = typer.Option(Path("configs/data/materialize_928.yaml")),
    subset_days: Optional[int] = typer.Option(None, help="Only use first N days (validation)"),
):
    """Window cleaned tracks and write pre-shuffled training shards."""
    from trackfm.config import MaterializeConfig, load_config
    from trackfm.datasets.materialize import materialize_dataset

    cfg = load_config(config, MaterializeConfig)
    materialize_dataset(cfg, subset_days=subset_days)


@app.command()
def audit(
    clean_dir: Path = typer.Option(Path("~/data/ais/clean")),
    raw_dir: Path = typer.Option(Path("~/data/ais/raw")),
    write_manifest: bool = typer.Option(True),
):
    """Audit the cleaned dataset: continuity, invariants, stats, MANIFEST."""
    from trackfm.data.audit import run_audit

    ok = run_audit(clean_dir.expanduser(), raw_dir.expanduser(), write_manifest)
    raise typer.Exit(0 if ok else 1)


@app.command()
def port_data(
    clean_dir: Path = typer.Option(Path("~/data/ais/clean")),
    out_dir: Path = typer.Option(Path("~/data/trackfm/ports/v2")),
    subset_days: Optional[int] = typer.Option(None, help="Only use first N cleaned days"),
    stride: int = typer.Option(64),
    ports_path: Optional[Path] = typer.Option(
        None, help="Reuse an existing ports.parquet (skip pass A discovery)"),
):
    """Discover ports/anchorages and materialize the voyage-labeled dataset."""
    import logging

    from trackfm.datasets.port_task import build_port_dataset

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    build_port_dataset(clean_dir, out_dir, stride=stride, subset_days=subset_days,
                       ports_path=ports_path)


@app.command()
def pretrain(
    config: Path = typer.Option(Path("configs/pretrain/xlarge.yaml")),
):
    """Pretrain the TrackFM foundation model."""
    from trackfm.config import PretrainConfig, load_config
    from trackfm.training.pretrain import run_pretraining

    cfg = load_config(config, PretrainConfig)
    run_pretraining(cfg)


@app.command()
def finetune(
    config: Path = typer.Option(..., help="FinetuneConfig YAML"),
):
    """Fine-tune a (pretrained or random-init) encoder on a downstream task."""
    import logging

    from trackfm.config import load_config
    from trackfm.training.finetune import FinetuneConfig, finetune_port_task

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config(config, FinetuneConfig)
    finetune_port_task(cfg)


@app.command()
def evaluate(
    checkpoint: Path = typer.Option(..., help="Model checkpoint"),
    config: Path = typer.Option(Path("configs/pretrain/xlarge.yaml")),
    split: str = typer.Option("val"),  # test retired from selection (audit F1)
    out: Optional[Path] = typer.Option(None, help="Write results JSON here"),
):
    """Per-horizon evaluation vs dead-reckoning / last-position baselines."""
    import json
    import logging

    from trackfm.config import PretrainConfig, load_config
    from trackfm.eval.forecast import evaluate_forecasting

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(config, PretrainConfig)
    results = evaluate_forecasting(checkpoint, cfg, split=split)
    if out:
        out.write_text(json.dumps(results, indent=2))
        typer.echo(f"written {out}")


if __name__ == "__main__":
    app()
