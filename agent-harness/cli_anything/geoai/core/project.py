"""Project management -- create, open, save, and inspect GeoAI workspaces.

A project is a JSON-serializable dict that tracks input files, models,
processing results, and metadata for a geospatial AI workflow.
"""

import copy
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional


def create_project(
    name: str = "untitled",
    description: str = "",
    crs: str = "EPSG:4326",
) -> Dict[str, Any]:
    """Create a new empty project.

    Args:
        name: Project name.
        description: Optional description.
        crs: Default coordinate reference system.

    Returns:
        Project dict.
    """
    now = datetime.now().isoformat()
    return {
        "name": name,
        "description": description,
        "created": now,
        "modified": now,
        "crs": crs,
        "files": [],
        "results": [],
        "models": [],
    }


def open_project(path: str) -> Dict[str, Any]:
    """Load a project from a JSON file.

    Args:
        path: Path to the project JSON file.

    Returns:
        Project dict.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid JSON or missing required keys.
    """
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Project file not found: {path}")

    with open(path, "r") as f:
        try:
            project = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in project file: {e}")

    required = {"name", "files"}
    missing = required - set(project.keys())
    if missing:
        raise ValueError(f"Project file missing required keys: {missing}")

    return project


def save_project(project: Dict[str, Any], path: str) -> str:
    """Save a project to a JSON file.

    Args:
        project: Project dict.
        path: Output file path.

    Returns:
        Absolute path to the saved file.
    """
    path = os.path.abspath(path)
    project["modified"] = datetime.now().isoformat()

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(project, f, indent=2, default=str)

    return path


def get_project_info(project: Dict[str, Any]) -> Dict[str, Any]:
    """Get summary information about a project.

    Args:
        project: Project dict.

    Returns:
        Summary dict with counts and metadata.
    """
    files = project.get("files", [])
    results = project.get("results", [])
    models = project.get("models", [])

    raster_count = sum(1 for f in files if f.get("type") == "raster")
    vector_count = sum(1 for f in files if f.get("type") == "vector")

    return {
        "name": project.get("name", "untitled"),
        "description": project.get("description", ""),
        "created": project.get("created", ""),
        "modified": project.get("modified", ""),
        "crs": project.get("crs", "EPSG:4326"),
        "file_count": len(files),
        "raster_count": raster_count,
        "vector_count": vector_count,
        "result_count": len(results),
        "model_count": len(models),
    }


def _next_id(project: Dict[str, Any], collection: str) -> int:
    """Get the next available ID for a collection.

    Args:
        project: Project dict.
        collection: Key name (files, results, models).

    Returns:
        Next integer ID.
    """
    items = project.get(collection, [])
    existing = [item.get("id", 0) for item in items]
    return max(existing, default=-1) + 1


def add_file(
    project: Dict[str, Any],
    path: str,
    file_type: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Add a file reference to the project.

    Args:
        project: Project dict.
        path: Path to the file.
        file_type: One of "raster", "vector", or auto-detected from extension.
        metadata: Optional metadata dict.

    Returns:
        The file entry that was added.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file type cannot be determined.
    """
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    if file_type is None:
        file_type = _detect_file_type(path)
    if file_type not in ("raster", "vector", "model", "config"):
        raise ValueError(
            f"Unknown file type: {file_type}. "
            "Use one of: raster, vector, model, config"
        )

    entry = {
        "id": _next_id(project, "files"),
        "path": path,
        "type": file_type,
        "added": datetime.now().isoformat(),
        "metadata": metadata or {},
    }

    if "files" not in project:
        project["files"] = []
    project["files"].append(entry)

    return entry


def remove_file(project: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Remove a file entry by index.

    Args:
        project: Project dict.
        index: Index in the files list.

    Returns:
        The removed file entry.

    Raises:
        IndexError: If index is out of range.
    """
    files = project.get("files", [])
    if index < 0 or index >= len(files):
        raise IndexError(f"File index {index} out of range (0-{len(files) - 1})")
    return files.pop(index)


def list_files(project: Dict[str, Any]) -> List[Dict[str, Any]]:
    """List all files in the project.

    Args:
        project: Project dict.

    Returns:
        List of file entries.
    """
    return project.get("files", [])


def add_result(
    project: Dict[str, Any],
    result_type: str,
    output_path: str,
    input_file_id: Optional[int] = None,
    model: str = "",
    parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Add a processing result to the project.

    Args:
        project: Project dict.
        result_type: Type of result (segmentation, detection, classification, change).
        output_path: Path to the output file.
        input_file_id: ID of the input file.
        model: Model name or path used.
        parameters: Processing parameters.

    Returns:
        The result entry.
    """
    entry = {
        "id": _next_id(project, "results"),
        "type": result_type,
        "input_file_id": input_file_id,
        "output_path": os.path.abspath(output_path),
        "model": model,
        "parameters": parameters or {},
        "created": datetime.now().isoformat(),
    }

    if "results" not in project:
        project["results"] = []
    project["results"].append(entry)

    return entry


def list_results(project: Dict[str, Any]) -> List[Dict[str, Any]]:
    """List all results in the project.

    Args:
        project: Project dict.

    Returns:
        List of result entries.
    """
    return project.get("results", [])


def _detect_file_type(path: str) -> str:
    """Detect file type from extension.

    Args:
        path: File path.

    Returns:
        File type string.

    Raises:
        ValueError: If extension is not recognized.
    """
    ext = os.path.splitext(path)[1].lower()
    raster_exts = {".tif", ".tiff", ".img", ".jp2", ".vrt", ".nc", ".hdf"}
    vector_exts = {
        ".geojson",
        ".json",
        ".shp",
        ".gpkg",
        ".parquet",
        ".geoparquet",
        ".fgb",
        ".kml",
    }
    model_exts = {".pth", ".pt", ".ckpt", ".onnx"}
    config_exts = {".yaml", ".yml"}

    if ext in raster_exts:
        return "raster"
    elif ext in vector_exts:
        return "vector"
    elif ext in model_exts:
        return "model"
    elif ext in config_exts:
        return "config"
    else:
        raise ValueError(
            f"Cannot detect file type for extension '{ext}'. "
            "Specify --type explicitly."
        )
