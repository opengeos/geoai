"""GeoAI CLI -- AI-powered geospatial analysis from the command line.

Provides a Click-based CLI with REPL support for image segmentation,
object detection, classification, change detection, data download,
and raster/vector operations.

Usage:
    cli-anything-geoai                          # Enter REPL
    cli-anything-geoai --json raster info f.tif # JSON output
    cli-anything-geoai segment sam image.tif -o mask.tif
"""

import json
import os
import shlex
import sys
from functools import wraps
from typing import Any, Dict, Optional

import click

from cli_anything.geoai.core import (
    change as change_mod,
    classify as classify_mod,
    data as data_mod,
    detect as detect_mod,
    project as project_mod,
    raster as raster_mod,
    segment as segment_mod,
    session as session_mod,
    vector as vector_mod,
)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_session: Optional[session_mod.Session] = None
_json_output: bool = False
_repl_mode: bool = False


def get_session() -> session_mod.Session:
    """Get or create the global session singleton.

    Returns:
        The Session instance.
    """
    global _session
    if _session is None:
        _session = session_mod.Session()
    return _session


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def output(data: Any, message: str = "") -> None:
    """Output data in JSON or human-readable format.

    Args:
        data: Data to output (dict, list, or scalar).
        message: Optional human-readable message for non-JSON mode.
    """
    if _json_output:
        click.echo(json.dumps(data, indent=2, default=str))
    else:
        if message:
            click.echo(message)
        if isinstance(data, dict):
            _print_dict(data)
        elif isinstance(data, list):
            _print_list(data)


def _print_dict(d: Dict[str, Any], indent: int = 0) -> None:
    """Pretty-print a dict with aligned keys.

    Args:
        d: Dict to print.
        indent: Left indentation spaces.
    """
    if not d:
        return
    max_key = max(len(str(k)) for k in d)
    prefix = " " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            click.echo(f"{prefix}{str(k):<{max_key}}:")
            _print_dict(v, indent + 2)
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            click.echo(f"{prefix}{str(k):<{max_key}}:")
            for item in v:
                _print_dict(item, indent + 2)
                click.echo()
        else:
            click.echo(f"{prefix}{str(k):<{max_key}}  {v}")


def _print_list(items: list) -> None:
    """Pretty-print a list of items.

    Args:
        items: List to print.
    """
    for item in items:
        if isinstance(item, dict):
            _print_dict(item)
            click.echo()
        else:
            click.echo(f"  {item}")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def handle_error(func):
    """Decorate a Click command to catch and format exceptions.

    In REPL mode, errors are displayed without exiting.
    In CLI mode, errors cause sys.exit(1).

    Args:
        func: The Click command function.

    Returns:
        Wrapped function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (
            ValueError,
            IndexError,
            FileNotFoundError,
            RuntimeError,
            OSError,
        ) as e:
            if _json_output:
                click.echo(
                    json.dumps(
                        {"error": str(e), "type": type(e).__name__},
                        indent=2,
                    )
                )
            else:
                click.echo(f"Error: {e}", err=True)
            if not _repl_mode:
                sys.exit(1)

    return wrapper


# ═══════════════════════════════════════════════════════════════════════════
# Main CLI group
# ═══════════════════════════════════════════════════════════════════════════


@click.group(invoke_without_command=True)
@click.option("--json", "use_json", is_flag=True, help="Output as JSON.")
@click.option(
    "--project",
    "project_path",
    type=str,
    default=None,
    help="Load a project file.",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "mps", "auto"]),
    default="auto",
    help="Compute device.",
)
@click.version_option(version="1.0.0", prog_name="cli-anything-geoai")
@click.pass_context
def cli(ctx, use_json, project_path, device):
    """GeoAI CLI -- AI-powered geospatial analysis.

    Run without arguments to enter the interactive REPL.
    Use --json for machine-readable output.
    """
    global _json_output
    _json_output = use_json

    if device and device != "auto":
        os.environ["GEOAI_DEVICE"] = device

    if project_path:
        sess = get_session()
        proj = project_mod.open_project(project_path)
        sess.set_project(proj, os.path.abspath(project_path))

    if ctx.invoked_subcommand is None:
        ctx.invoke(repl, project_path=project_path)


# ═══════════════════════════════════════════════════════════════════════════
# Project commands
# ═══════════════════════════════════════════════════════════════════════════


@cli.group()
def project():
    """Project management -- create, open, save workspaces."""
    pass


@project.command("new")
@click.option("--name", "-n", default="untitled", help="Project name.")
@click.option("--description", "-d", default="", help="Description.")
@click.option("--crs", default="EPSG:4326", help="Default CRS.")
@click.option("--output", "-o", default=None, help="Save to file immediately.")
@handle_error
def project_new(name, description, crs, output):
    """Create a new GeoAI project."""
    proj = project_mod.create_project(name=name, description=description, crs=crs)
    sess = get_session()
    sess.set_project(proj, output)

    if output:
        project_mod.save_project(proj, output)

    data = project_mod.get_project_info(proj)
    globals()["output"](data, f"Created project: {name}")


@project.command("open")
@click.argument("path", type=click.Path(exists=True))
@handle_error
def project_open(path):
    """Open an existing project file."""
    proj = project_mod.open_project(path)
    sess = get_session()
    sess.set_project(proj, os.path.abspath(path))
    data = project_mod.get_project_info(proj)
    output(data, f"Opened project: {proj.get('name', 'untitled')}")


@project.command("save")
@click.option(
    "--output", "-o", "output_path", default=None, help="Save path (default: original path)."
)
@handle_error
def project_save(output_path=None):
    """Save the current project."""
    sess = get_session()
    path = sess.save_session(output_path)
    globals()["output"](
        {"saved": path, "project": sess.project.get("name", "untitled")},
        f"Project saved to: {path}",
    )


@project.command("info")
@handle_error
def project_info():
    """Show project information."""
    sess = get_session()
    proj = sess.get_project()
    data = project_mod.get_project_info(proj)
    output(data)


@project.command("add-file")
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--type",
    "file_type",
    type=click.Choice(["raster", "vector", "model", "config"]),
    default=None,
    help="File type (auto-detected if omitted).",
)
@handle_error
def project_add_file(path, file_type):
    """Add a file to the project."""
    sess = get_session()
    sess.snapshot("Add file")
    proj = sess.get_project()
    entry = project_mod.add_file(proj, path, file_type=file_type)
    output(entry, f"Added {entry['type']}: {os.path.basename(path)}")


@project.command("list-files")
@handle_error
def project_list_files():
    """List all files in the project."""
    sess = get_session()
    proj = sess.get_project()
    files = project_mod.list_files(proj)
    if not files:
        output([], "No files in project.")
    else:
        output(files)


@project.command("list-results")
@handle_error
def project_list_results():
    """List all processing results in the project."""
    sess = get_session()
    proj = sess.get_project()
    results = project_mod.list_results(proj)
    if not results:
        output([], "No results in project.")
    else:
        output(results)


# ═══════════════════════════════════════════════════════════════════════════
# Raster commands
# ═══════════════════════════════════════════════════════════════════════════


@cli.group()
def raster():
    """Raster file inspection and operations."""
    pass


@raster.command("info")
@click.argument("path", type=click.Path(exists=True))
@handle_error
def raster_info(path):
    """Show metadata for a raster file."""
    data = raster_mod.get_raster_info(path)
    output(data)


@raster.command("stats")
@click.argument("path", type=click.Path(exists=True))
@click.option("--band", "-b", type=int, default=1, help="Band number (1-indexed).")
@handle_error
def raster_stats(path, band):
    """Compute statistics for a raster band."""
    data = raster_mod.get_raster_stats(path, band=band)
    output(data)


@raster.command("vectorize")
@click.argument("path", type=click.Path(exists=True))
@click.option("--output-path", "-o", required=True, help="Output vector file path.")
@click.option("--simplify", type=float, default=None, help="Simplification tolerance.")
@handle_error
def raster_vectorize(path, output_path, simplify):
    """Convert a raster to vector polygons."""
    data = raster_mod.vectorize_raster(
        path, output_path, simplify_tolerance=simplify
    )
    globals()["output"](data, f"Vectorized to: {output_path}")


@raster.command("tile")
@click.argument("path", type=click.Path(exists=True))
@click.option("--output-dir", "-o", required=True, help="Output directory for tiles.")
@click.option("--tile-size", "-s", type=int, default=512, help="Tile size in pixels.")
@click.option("--overlap", type=int, default=0, help="Overlap between tiles in pixels.")
@handle_error
def raster_tile(path, output_dir, tile_size, overlap):
    """Split a raster into tiles."""
    data = raster_mod.tile_raster(
        path, output_dir, tile_size=tile_size, overlap=overlap
    )
    output(data, f"Created {data['tile_count']} tiles in: {output_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# Vector commands
# ═══════════════════════════════════════════════════════════════════════════


@cli.group()
def vector():
    """Vector file inspection and operations."""
    pass


@vector.command("info")
@click.argument("path", type=click.Path(exists=True))
@handle_error
def vector_info(path):
    """Show metadata for a vector file."""
    data = vector_mod.get_vector_info(path)
    output(data)


@vector.command("rasterize")
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--template",
    "-t",
    required=True,
    type=click.Path(exists=True),
    help="Template raster for extent/resolution.",
)
@click.option("--output", "-o", required=True, help="Output raster file path.")
@click.option("--attribute", "-a", default=None, help="Attribute column to burn.")
@handle_error
def vector_rasterize(path, template, output_path, attribute):
    """Convert a vector file to raster."""
    data = vector_mod.rasterize_vector(path, template, output_path, attribute=attribute)
    globals()["output"](data, f"Rasterized to: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Data commands
# ═══════════════════════════════════════════════════════════════════════════


@cli.group()
def data():
    """Data discovery, download, and preparation."""
    pass


@data.command("search")
@click.option("--bbox", required=True, help="Bounding box: minx,miny,maxx,maxy.")
@click.option(
    "--collection", "-c", default="sentinel-2-l2a", help="STAC collection ID."
)
@click.option("--start-date", default=None, help="Start date (YYYY-MM-DD).")
@click.option("--end-date", default=None, help="End date (YYYY-MM-DD).")
@click.option("--max-items", type=int, default=10, help="Maximum results.")
@handle_error
def data_search(bbox, collection, start_date, end_date, max_items):
    """Search for satellite imagery via STAC API."""
    bbox_tuple = data_mod.parse_bbox(bbox)
    result = data_mod.search_stac(
        bbox=bbox_tuple,
        collection=collection,
        start_date=start_date,
        end_date=end_date,
        max_items=max_items,
    )
    output(result, f"Found {result['total_results']} items")


@data.command("download")
@click.argument("source", type=click.Choice(["naip", "overture"]))
@click.option("--bbox", required=True, help="Bounding box: minx,miny,maxx,maxy.")
@click.option("--output", "-o", "output_path", required=True, help="Output file/directory.")
@click.option("--year", type=int, default=None, help="Year (for NAIP).")
@handle_error
def data_download(source, bbox, output_path, year):
    """Download geospatial data from supported sources."""
    bbox_tuple = data_mod.parse_bbox(bbox)

    if source == "naip":
        result = data_mod.download_naip(bbox=bbox_tuple, output=output_path, year=year)
    elif source == "overture":
        result = data_mod.download_overture(bbox=bbox_tuple, output=output_path)
    else:
        raise ValueError(f"Unsupported source: {source}")

    globals()["output"](result, f"Downloaded from {source}")


@data.command("sources")
@handle_error
def data_sources():
    """List available data sources."""
    sources = data_mod.list_sources()
    output(sources)


# ═══════════════════════════════════════════════════════════════════════════
# Segment commands
# ═══════════════════════════════════════════════════════════════════════════


@cli.group()
def segment():
    """Image segmentation -- SAM, GroundedSAM, semantic segmentation."""
    pass


@segment.command("sam")
@click.argument("raster", type=click.Path(exists=True))
@click.option("--output", "-o", "output_path", required=True, help="Output mask file path.")
@click.option("--model", "-m", default="facebook/sam-vit-huge", help="SAM model ID.")
@click.option("--no-automatic", is_flag=True, help="Disable automatic mask generation.")
@click.option("--no-foreground", is_flag=True, help="Include background segments.")
@click.option("--no-unique", is_flag=True, help="Do not assign unique labels.")
@handle_error
def segment_sam(raster, output_path, model, no_automatic, no_foreground, no_unique):
    """Run SAM (Segment Anything Model) segmentation."""
    result = segment_mod.run_sam(
        raster=raster,
        output=output_path,
        model=model,
        automatic=not no_automatic,
        foreground=not no_foreground,
        unique=not no_unique,
    )
    globals()["output"](result, f"SAM segmentation complete: {output_path}")


@segment.command("grounded-sam")
@click.argument("raster", type=click.Path(exists=True))
@click.option("--output-path", "-o", required=True, help="Output mask file path.")
@click.option(
    "--prompt", "-p", required=True, help="Text prompt describing objects to segment."
)
@click.option(
    "--detector-model",
    default="IDEA-Research/grounding-dino-tiny",
    help="Grounding DINO model.",
)
@click.option("--segmenter-model", default="facebook/sam-vit-huge", help="SAM model.")
@click.option("--tile-size", type=int, default=1024, help="Processing tile size.")
@handle_error
def segment_grounded_sam(
    raster, output_path, prompt, detector_model, segmenter_model, tile_size
):
    """Run GroundedSAM text-prompted segmentation."""
    result = segment_mod.run_grounded_sam(
        raster=raster,
        output=output_path,
        prompt=prompt,
        detector_model=detector_model,
        segmenter_model=segmenter_model,
        tile_size=tile_size,
    )
    globals()["output"](result, f"GroundedSAM complete: {output_path}")


@segment.command("semantic")
@click.argument("raster", type=click.Path(exists=True))
@click.option(
    "--model",
    "-m",
    required=True,
    type=click.Path(exists=True),
    help="Path to trained segmentation model.",
)
@click.option("--output", "-o", "output_path", required=True, help="Output segmentation raster.")
@click.option("--num-classes", "-n", type=int, default=2, help="Number of classes.")
@click.option("--chip-size", type=int, default=512, help="Processing chip size.")
@click.option(
    "--overlap", type=float, default=0.25, help="Overlap fraction between chips."
)
@handle_error
def segment_semantic(raster, model, output_path, num_classes, chip_size, overlap):
    """Run semantic segmentation with a trained model."""
    result = segment_mod.run_semantic_segmentation(
        raster=raster,
        model_path=model,
        output=output_path,
        num_classes=num_classes,
        chip_size=chip_size,
        overlap=overlap,
    )
    globals()["output"](result, f"Semantic segmentation complete: {output_path}")


@segment.command("train")
@click.option(
    "--images",
    "-i",
    required=True,
    type=click.Path(exists=True),
    help="Training images directory.",
)
@click.option(
    "--labels",
    "-l",
    required=True,
    type=click.Path(exists=True),
    help="Training labels directory.",
)
@click.option("--output-dir", "-o", required=True, help="Output directory for model.")
@click.option("--arch", default="unet", help="Architecture (unet, deeplabv3, fpn).")
@click.option("--backbone", default="resnet50", help="Encoder backbone.")
@click.option("--num-classes", "-n", type=int, default=2, help="Number of classes.")
@click.option("--in-channels", type=int, default=4, help="Input channels.")
@click.option("--epochs", type=int, default=20, help="Training epochs.")
@click.option("--batch-size", type=int, default=8, help="Batch size.")
@click.option("--lr", type=float, default=1e-4, help="Learning rate.")
@click.option("--loss", default="ce", help="Loss function (ce, jaccard, focal, dice).")
@handle_error
def segment_train(
    images,
    labels,
    output_dir,
    arch,
    backbone,
    num_classes,
    in_channels,
    epochs,
    batch_size,
    lr,
    loss,
):
    """Train a semantic segmentation model."""
    result = segment_mod.train_segmentation(
        image_root=images,
        label_root=labels,
        output_dir=output_dir,
        architecture=arch,
        backbone=backbone,
        num_classes=num_classes,
        in_channels=in_channels,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        loss=loss,
    )
    output(result, f"Training complete. Model saved to: {output_dir}")


@segment.command("list-models")
@handle_error
def segment_list_models():
    """List available SAM models."""
    models = segment_mod.list_sam_models()
    output(models)


@segment.command("list-architectures")
@handle_error
def segment_list_architectures():
    """List available segmentation architectures."""
    archs = segment_mod.list_architectures()
    output(archs)


# ═══════════════════════════════════════════════════════════════════════════
# Detect commands
# ═══════════════════════════════════════════════════════════════════════════


@cli.group()
def detect():
    """Object detection -- run and train detectors."""
    pass


@detect.command("run")
@click.argument("raster", type=click.Path(exists=True))
@click.option(
    "--model",
    "-m",
    required=True,
    type=click.Path(exists=True),
    help="Path to trained detection model.",
)
@click.option(
    "--num-classes",
    "-n",
    type=int,
    required=True,
    help="Number of classes (including background).",
)
@click.option(
    "--output-vector", "-ov", default=None, help="Output vector file for detections."
)
@click.option(
    "--output-raster",
    "-or",
    default=None,
    help="Output raster file for detection mask.",
)
@click.option("--confidence", type=float, default=0.5, help="Confidence threshold.")
@click.option("--chip-size", type=int, default=512, help="Processing chip size.")
@click.option("--overlap", type=float, default=0.25, help="Overlap between chips.")
@handle_error
def detect_run(
    raster,
    model,
    num_classes,
    output_vector,
    output_raster,
    confidence,
    chip_size,
    overlap,
):
    """Run object detection on a raster image."""
    result = detect_mod.run_detection(
        raster=raster,
        model_path=model,
        num_classes=num_classes,
        output_vector=output_vector,
        output_raster=output_raster,
        confidence_threshold=confidence,
        chip_size=chip_size,
        overlap=overlap,
    )
    output(result, "Detection complete.")


@detect.command("train")
@click.option(
    "--images",
    "-i",
    required=True,
    type=click.Path(exists=True),
    help="Training images directory.",
)
@click.option(
    "--labels",
    "-l",
    required=True,
    type=click.Path(exists=True),
    help="Training labels directory.",
)
@click.option("--output-dir", "-o", required=True, help="Output directory.")
@click.option(
    "--model", "-m", default="maskrcnn_resnet50_fpn", help="Model architecture."
)
@click.option(
    "--num-classes",
    "-n",
    type=int,
    required=True,
    help="Number of classes (including background).",
)
@click.option(
    "--format",
    "input_format",
    default="directory",
    type=click.Choice(["directory", "coco", "yolo"]),
    help="Label format.",
)
@click.option("--epochs", type=int, default=30, help="Training epochs.")
@click.option("--batch-size", type=int, default=4, help="Batch size.")
@click.option("--lr", type=float, default=0.005, help="Learning rate.")
@handle_error
def detect_train(
    images, labels, output_dir, model, num_classes, input_format, epochs, batch_size, lr
):
    """Train an object detection model."""
    result = detect_mod.train_detector(
        images_dir=images,
        labels_dir=labels,
        output_dir=output_dir,
        model_name=model,
        num_classes=num_classes,
        input_format=input_format,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
    )
    output(result, f"Training complete. Model saved to: {output_dir}")


@detect.command("list-models")
@handle_error
def detect_list_models():
    """List available detection model architectures."""
    models = detect_mod.list_models()
    output(models)


# ═══════════════════════════════════════════════════════════════════════════
# Classify commands
# ═══════════════════════════════════════════════════════════════════════════


@cli.group()
def classify():
    """Image classification -- train and predict."""
    pass


@classify.command("train")
@click.option(
    "--train-dir",
    "-t",
    required=True,
    type=click.Path(exists=True),
    help="Training data directory.",
)
@click.option(
    "--val-dir",
    "-v",
    default=None,
    type=click.Path(exists=True),
    help="Validation data directory.",
)
@click.option("--output-dir", "-o", required=True, help="Output directory.")
@click.option("--model", "-m", default="resnet50", help="TIMM model name.")
@click.option("--epochs", type=int, default=30, help="Training epochs.")
@click.option("--batch-size", type=int, default=32, help="Batch size.")
@click.option("--lr", type=float, default=1e-3, help="Learning rate.")
@click.option("--image-size", type=int, default=224, help="Input image size.")
@click.option("--in-channels", type=int, default=3, help="Input channels.")
@handle_error
def classify_train(
    train_dir,
    val_dir,
    output_dir,
    model,
    epochs,
    batch_size,
    lr,
    image_size,
    in_channels,
):
    """Train an image classification model."""
    result = classify_mod.train_classifier(
        train_dir=train_dir,
        val_dir=val_dir,
        output_dir=output_dir,
        model_name=model,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        image_size=image_size,
        in_channels=in_channels,
    )
    output(result, f"Training complete. Model saved to: {output_dir}")


@classify.command("predict")
@click.argument("image", type=click.Path(exists=True))
@click.option(
    "--model",
    "-m",
    required=True,
    type=click.Path(exists=True),
    help="Path to trained model.",
)
@click.option("--num-classes", "-n", type=int, default=None, help="Number of classes.")
@click.option("--image-size", type=int, default=224, help="Input image size.")
@handle_error
def classify_predict(image, model, num_classes, image_size):
    """Classify an image with a trained model."""
    result = classify_mod.predict_classification(
        image_path=image,
        model_path=model,
        num_classes=num_classes,
        image_size=image_size,
    )
    output(result, "Classification result:")


# ═══════════════════════════════════════════════════════════════════════════
# Change detection commands
# ═══════════════════════════════════════════════════════════════════════════


@cli.group()
def change():
    """Change detection between temporal images."""
    pass


@change.command("detect")
@click.argument("image1", type=click.Path(exists=True))
@click.argument("image2", type=click.Path(exists=True))
@click.option("--output", "-o", "output_path", required=True, help="Output change map path.")
@click.option(
    "--method",
    default="changestar",
    type=click.Choice(["changestar", "anychange"]),
    help="Detection method.",
)
@click.option(
    "--confidence", type=int, default=155, help="Confidence threshold (0-255)."
)
@click.option(
    "--min-area", type=int, default=0, help="Min area for change regions (pixels)."
)
@handle_error
def change_detect(image1, image2, output_path, method, confidence, min_area):
    """Detect changes between two temporal images."""
    result = change_mod.detect_changes(
        image1=image1,
        image2=image2,
        output=output_path,
        method=method,
        confidence_threshold=confidence,
        min_area=min_area,
    )
    globals()["output"](result, f"Change detection complete: {output_path}")


@change.command("list-methods")
@handle_error
def change_list_methods():
    """List available change detection methods."""
    methods = change_mod.list_methods()
    output(methods)


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline commands
# ═══════════════════════════════════════════════════════════════════════════


@cli.group()
def pipeline():
    """Batch processing pipelines."""
    pass


@pipeline.command("run")
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--input-dir", "-i", default=None, help="Input directory.")
@click.option("--output-dir", "-o", default=None, help="Output directory.")
@click.option("--max-workers", "-w", type=int, default=None, help="Parallel workers.")
@click.option("--checkpoint-dir", "-c", default=None, help="Checkpoint directory.")
@handle_error
def pipeline_run(config_path, input_dir, output_dir, max_workers, checkpoint_dir):
    """Run a pipeline from a JSON or YAML config file."""
    from geoai.pipeline import load_pipeline

    overrides = {}
    if max_workers is not None:
        overrides["max_workers"] = max_workers
    if checkpoint_dir is not None:
        overrides["checkpoint_dir"] = checkpoint_dir

    pipe = load_pipeline(config_path, **overrides)

    kwargs = {}
    if input_dir:
        kwargs["input_dir"] = input_dir
    if output_dir:
        kwargs["output_dir"] = output_dir

    result = pipe.run(**kwargs)
    summary = result.summary

    data = {
        "completed": summary["completed"],
        "failed": summary["failed"],
        "skipped": summary["skipped"],
        "duration_seconds": summary["total_duration"],
    }
    if result.checkpoint_path:
        data["checkpoint"] = result.checkpoint_path

    output(data, "Pipeline complete.")


@pipeline.command("show")
@click.argument("config_path", type=click.Path(exists=True))
@handle_error
def pipeline_show(config_path):
    """Display a pipeline configuration."""
    from geoai.pipeline import load_pipeline

    pipe = load_pipeline(config_path)
    data = {
        "name": pipe.name,
        "max_workers": pipe.max_workers,
        "on_error": pipe.on_error.value,
        "steps": [str(s) for s in pipe.steps],
    }
    output(data)


# ═══════════════════════════════════════════════════════════════════════════
# Session commands
# ═══════════════════════════════════════════════════════════════════════════


@cli.group()
def session():
    """Session management -- status, undo, redo, history."""
    pass


@session.command("status")
@handle_error
def session_status():
    """Show current session status."""
    sess = get_session()
    data = sess.get_status()
    output(data)


@session.command("undo")
@handle_error
def session_undo():
    """Undo the last operation."""
    sess = get_session()
    desc = sess.undo()
    output(
        {"undone": desc, **sess.get_status()},
        f"Undone: {desc}" if desc else "Undone.",
    )


@session.command("redo")
@handle_error
def session_redo():
    """Redo a previously undone operation."""
    sess = get_session()
    desc = sess.redo()
    output(
        {"redone": desc, **sess.get_status()},
        f"Redone: {desc}" if desc else "Redone.",
    )


@session.command("history")
@handle_error
def session_history():
    """Show operation history."""
    sess = get_session()
    hist = sess.history()
    if not hist:
        output([], "No history.")
    else:
        output(hist)


# ═══════════════════════════════════════════════════════════════════════════
# System info command
# ═══════════════════════════════════════════════════════════════════════════


@cli.command("system-info")
@handle_error
def system_info():
    """Show system and dependency information."""
    from cli_anything.geoai.utils.geoai_backend import get_system_info

    data = get_system_info()
    output(data)


# ═══════════════════════════════════════════════════════════════════════════
# REPL command
# ═══════════════════════════════════════════════════════════════════════════


_REPL_COMMANDS = {
    "project": "new | open | save | info | add-file | list-files | list-results",
    "raster": "info | stats | vectorize | tile",
    "vector": "info | rasterize",
    "data": "search | download | sources",
    "segment": "sam | grounded-sam | semantic | train | list-models | list-architectures",
    "detect": "run | train | list-models",
    "classify": "train | predict",
    "change": "detect | list-methods",
    "pipeline": "run | show",
    "session": "status | undo | redo | history",
    "system-info": "show system and dependency information",
    "help": "show this help",
    "quit": "exit REPL",
}


@cli.command()
@click.option("--project-path", default=None, hidden=True)
@handle_error
def repl(project_path):
    """Start the interactive REPL session."""
    from cli_anything.geoai.utils.repl_skin import ReplSkin

    global _repl_mode
    _repl_mode = True

    skin = ReplSkin("geoai", version="1.0.0")
    skin.print_banner()

    pt_session = skin.create_prompt_session()

    while True:
        try:
            sess = get_session()
            project_name = ""
            modified = False
            if sess.has_project():
                name = sess.project.get("name", "untitled")
                project_name = os.path.basename(sess.project_path or name)
                modified = sess._modified

            line = skin.get_input(
                pt_session, project_name=project_name, modified=modified
            ).strip()

            if not line:
                continue
            if line.lower() in ("quit", "exit", "q"):
                break
            if line.lower() == "help":
                skin.help(_REPL_COMMANDS)
                continue

            try:
                args = shlex.split(line)
            except ValueError:
                args = line.split()

            try:
                cli.main(args, standalone_mode=False)
            except SystemExit:
                pass
            except click.exceptions.UsageError as e:
                skin.warning(f"Usage: {e}")
            except Exception as e:
                skin.error(str(e))

        except (EOFError, KeyboardInterrupt):
            break

    skin.print_goodbye()
    _repl_mode = False


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════


def main():
    """Entry point for console_scripts."""
    cli()


if __name__ == "__main__":
    main()
