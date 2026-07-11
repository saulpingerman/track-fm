"""TrackFM command-line interface.

Subcommands mirror the project stages:
  download     Fetch raw DMA AIS zips
  clean        Run the cleaning pipeline (raw zips -> partitioned parquet)
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
def port_data(
    clean_dir: Path = typer.Option(Path("~/data/ais/clean")),
    out_dir: Path = typer.Option(Path("~/data/trackfm/ports/v1")),
    subset_days: Optional[int] = typer.Option(None, help="Only use first N cleaned days"),
    stride: int = typer.Option(64),
):
    """Discover ports, label tracks, and materialize the port-task dataset."""
    import logging

    import polars as pl

    from trackfm.datasets.port_task import build_port_task_dataset
    from trackfm.datasets.ports import (PortLabelConfig, discover_ports,
                                        extract_track_endpoints, label_tracks)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    clean_dir = clean_dir.expanduser()
    out_dir = out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    from trackfm.datasets.materialize import list_day_partitions
    days = list_day_partitions(clean_dir)
    if subset_days:
        days = days[:subset_days]

    cfg = PortLabelConfig()
    tracks = extract_track_endpoints(clean_dir, day_files=days,
                                     min_positions=cfg.min_track_positions)
    ports = discover_ports(tracks, cfg)
    ports.write_parquet(out_dir / "ports.parquet")
    labeled = label_tracks(tracks, ports, cfg)
    labeled.write_parquet(out_dir / "labeled_tracks.parquet")
    build_port_task_dataset(clean_dir, labeled, out_dir,
                            stride=stride, subset_days=subset_days)


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
    config: Path = typer.Option(..., help="Downstream task config YAML"),
):
    """Fine-tune a pretrained backbone on a downstream task."""
    typer.echo("Downstream fine-tuning: see experiments/anomaly and experiments/vessel_class")
    raise typer.Exit(1)


@app.command()
def evaluate(
    checkpoint: Path = typer.Option(..., help="Model checkpoint"),
    config: Path = typer.Option(Path("configs/pretrain/xlarge.yaml")),
):
    """Evaluate a checkpoint against dead-reckoning / last-position baselines."""
    typer.echo("Evaluation entry point: implemented with the trainer in Phase 2.")
    raise typer.Exit(1)


if __name__ == "__main__":
    app()
