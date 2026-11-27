"""Helpers for the Segment Anything panel's saved outputs.

These are kept free of Qt widgets so they can be unit tested without a QGIS
session. Only :func:`add_mask_attributes` touches ``qgis.core``, and it
imports it lazily.
"""

from __future__ import annotations

import math
import os
import re
import tempfile
from typing import Callable, Dict, Iterable, Mapping, Optional

# Columns that hold the mask's raster value in the vectorized output, in the
# order they are tried. ``raster_to_vector`` and ``smooth_vector`` (Simple and
# Smooth modes) write ``class``; ``orthogonalize`` (Regularize mode) writes
# ``value``.
MASK_ID_FIELDS = ("class", "value")

# Shapefile (dBase) field names are limited to 10 characters and text values
# to 254 characters.
SHAPEFILE_FIELD_NAME_MAX = 10
SHAPEFILE_STRING_MAX = 254

DEFAULT_OUTPUT_SUFFIX = "_sam_masks"

HF_SAM3_URL = "https://huggingface.co/facebook/sam3"
HF_TOKENS_URL = "https://huggingface.co/settings/tokens"

# An error counts as a Hugging Face access problem only when it mentions both
# Hugging Face (or the SAM 3 repo) and an authentication failure, so that
# unrelated 401/403 errors (for example from a proxy) do not trigger the
# SAM 3 help dialog.
_HF_CONTEXT_MARKERS = (
    "huggingface",
    "hugging face",
    "hf.co",
    "hf_hub",
    "facebook/sam3",
    "gatedrepoerror",
)
_HF_AUTH_MARKERS = (
    "gated repo",
    "gatedrepo",
    "gated repository",
    "is restricted",
    "401 client error",
    "403 client error",
    "repository not found",
    "make sure you are authenticated",
    "you must have access",
    "cannot access gated",
    "token is required",
    "not logged in",
    "invalid user token",
    "unauthorized",
)

# ``GPKG:<path>.gpkg[:<layer>]``: the path ends at a ``.gpkg`` that is
# followed by the layer separator or the end of the string.
_GPKG_PREFIXED = re.compile(r"^(.*?\.gpkg)(?::[^/\\]*)?$", re.IGNORECASE)


def output_extension(format_text: str) -> str:
    """Map an Output-tab format label to a file extension.

    Args:
        format_text: Text of the output format combo box, for example
            ``"Vector (GeoPackage)"``.

    Returns:
        The extension including the leading dot (``.tif``, ``.geojson``,
        ``.gpkg`` or ``.shp``).
    """
    if "GeoJSON" in format_text:
        return ".geojson"
    if "GeoPackage" in format_text:
        return ".gpkg"
    if "Shapefile" in format_text:
        return ".shp"
    return ".tif"


def source_file_path(path: Optional[str]) -> Optional[str]:
    """Strip provider decorations from a QGIS layer source to get a file path.

    Handles ``GPKG:/path/file.gpkg:layer`` and ``/path/file.gpkg|layername=x``
    style sources.

    Args:
        path: A file path or QGIS layer source string.

    Returns:
        The bare file path, or None if ``path`` is empty.
    """
    if not path:
        return None
    path = str(path).split("|", 1)[0]
    if path.upper().startswith("GPKG:"):
        rest = path[5:]
        match = _GPKG_PREFIXED.match(rest)
        path = match.group(1) if match else rest
    return path


def suggested_output_name(image_path: Optional[str], ext: str) -> str:
    """Build the base name suggested for saved masks.

    Args:
        image_path: Path of the segmented image (may be None).
        ext: Output extension including the leading dot.

    Returns:
        ``<image_stem>_sam_masks<ext>``, or ``sam_masks<ext>`` without an image.
    """
    base = source_file_path(image_path)
    stem = os.path.splitext(os.path.basename(base))[0] if base else ""
    if not stem:
        return f"sam_masks{ext}"
    return f"{stem}{DEFAULT_OUTPUT_SUFFIX}{ext}"


def unique_path(path: str, exists: Callable[[str], bool] = os.path.exists) -> str:
    """Return ``path`` or the first ``<stem>_N<ext>`` variant that does not exist.

    Args:
        path: The preferred path.
        exists: Existence check, injectable for tests.

    Returns:
        A path that does not exist yet.
    """
    if not exists(path):
        return path
    stem, ext = os.path.splitext(path)
    n = 1
    while exists(f"{stem}_{n}{ext}"):
        n += 1
    return f"{stem}_{n}{ext}"


def default_output_path(
    image_path: Optional[str],
    ext: str,
    exists: Callable[[str], bool] = os.path.exists,
    is_writable_dir: Optional[Callable[[str], bool]] = None,
) -> Optional[str]:
    """Pick a non-overwriting output path next to the source image.

    Args:
        image_path: Path of the segmented image (a QGIS source string is
            accepted too).
        ext: Output extension including the leading dot.
        exists: Existence check, injectable for tests.
        is_writable_dir: Directory writability check, injectable for tests.
            Defaults to ``os.path.isdir`` and ``os.access(..., W_OK)``.

    Returns:
        ``<image_dir>/<image_stem>_sam_masks[_N]<ext>``, or None when the image
        is not a local file or its directory is not writable (the caller then
        falls back to a temporary file).
    """
    base = source_file_path(image_path)
    if not base or not os.path.isabs(base) or not exists(base):
        return None
    directory = os.path.dirname(base)
    if is_writable_dir is None:

        def is_writable_dir(d: str) -> bool:
            return os.path.isdir(d) and os.access(d, os.W_OK)

    if not directory or not is_writable_dir(directory):
        return None
    return unique_path(
        os.path.join(directory, suggested_output_name(base, ext)), exists=exists
    )


def temp_output_path(ext: str) -> str:
    """Create a temporary output path for the given extension.

    Shapefiles get their own temporary folder so their sidecar files stay
    together.

    Args:
        ext: Output extension including the leading dot.

    Returns:
        Path to a new, empty temporary file (or a not-yet-created ``.shp`` in
        a new temporary folder).
    """
    if ext == ".shp":
        return os.path.join(tempfile.mkdtemp(prefix="samgeo_"), "masks.shp")
    handle = tempfile.NamedTemporaryFile(suffix=ext, delete=False, prefix="samgeo_")
    handle.close()
    return handle.name


def count_raster_labels(path: str, unique: bool) -> Optional[int]:
    """Count distinct non-zero labels in a mask raster, block by block.

    Memory grows with the number of objects, not with the raster size.

    Args:
        path: Path to the mask raster.
        unique: Whether each object has its own label.

    Returns:
        The label count, or None for binary masks or if the raster cannot be
        read (for example when rasterio is not importable).
    """
    if not unique:
        return None
    try:
        import numpy as np
        import rasterio

        labels = set()
        with rasterio.open(path) as src:
            for _, window in src.block_windows(1):
                labels.update(np.unique(src.read(1, window=window)).tolist())
        labels.discard(0)
        return len(labels)
    except Exception:
        return None


def field_name(name: str, is_shapefile: bool) -> str:
    """Return a field name valid for the output format.

    Args:
        name: Desired field name.
        is_shapefile: Whether the output is an ESRI Shapefile.

    Returns:
        The name, truncated to 10 characters for shapefiles.
    """
    return name[:SHAPEFILE_FIELD_NAME_MAX] if is_shapefile else name


def find_mask_id_field(field_names: Iterable[str]) -> Optional[str]:
    """Find the column that stores the mask's raster value.

    Args:
        field_names: Field names of the vectorized layer.

    Returns:
        The matching field name as spelled in the layer (``class`` or
        ``value``, case-insensitive), or None.
    """
    by_lower = {str(n).lower(): str(n) for n in field_names}
    for candidate in MASK_ID_FIELDS:
        if candidate in by_lower:
            return by_lower[candidate]
    return None


def normalize_mask_scores(scores: Optional[Mapping]) -> Dict[int, float]:
    """Convert a mask-score mapping to ``{int mask id: float score}``.

    Scores arrive with string keys after a JSON round trip from the SamGeo
    worker process; entries that cannot be converted, and non-finite scores
    (NaN, infinity), are dropped.

    Args:
        scores: Mapping of mask id to score, or None.

    Returns:
        A new dict with integer keys and float values (empty if none usable).
    """
    result: Dict[int, float] = {}
    if not scores:
        return result
    try:
        items = list(scores.items())
    except AttributeError:
        return result
    for key, value in items:
        try:
            mask_id = int(float(key))
            score = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(score):
            continue
        result[mask_id] = score
    return result


def mask_id_from_value(value) -> Optional[int]:
    """Convert an attribute value read from the mask-id column to an int.

    Args:
        value: The attribute value (int, float, str or a NULL variant).

    Returns:
        The integer mask id, or None if the value is missing or not numeric.
    """
    if value is None:
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError, OverflowError):
        return None


def format_prompt_label(kind: str, value: Optional[str] = None) -> str:
    """Describe the prompt that produced the masks, for the ``prompt`` field.

    Args:
        kind: Prompt type, for example ``"text"``, ``"point"``, ``"box"`` or
            ``"batch points"``.
        value: Optional prompt value, for example the text prompt.

    Returns:
        ``"<kind>: <value>"`` when a value is given, else ``"<kind>"``.
    """
    value = (value or "").strip()
    return f"{kind}: {value}" if value else kind


def redact(text: Optional[str], secret: Optional[str]) -> str:
    """Remove a secret (such as an access token) from a message.

    Args:
        text: The message.
        secret: The secret to hide.

    Returns:
        ``text`` with every occurrence of ``secret`` replaced by ``***``.
    """
    text = "" if text is None else str(text)
    if secret:
        secret = str(secret).strip()
        if secret:
            text = text.replace(secret, "***")
    return text


def is_hf_access_error(message: Optional[str]) -> bool:
    """Tell whether a model-loading error means Hugging Face denied access.

    Args:
        message: The error message.

    Returns:
        True for gated-repo, 401/403 and missing-token errors that mention
        Hugging Face or the SAM 3 repository.
    """
    lower = (message or "").lower()
    return any(marker in lower for marker in _HF_CONTEXT_MARKERS) and any(
        marker in lower for marker in _HF_AUTH_MARKERS
    )


def hf_access_help(message: Optional[str]) -> Optional[str]:
    """Build an actionable explanation for a Hugging Face access error.

    Args:
        message: The original model-loading error message.

    Returns:
        A user-facing explanation, or None if ``message`` is not an access
        error.
    """
    if not is_hf_access_error(message):
        return None
    return (
        "SAM 3 is hosted in a gated Hugging Face repository and the plugin "
        "could not download it.\n\n"
        f"1. Request access at {HF_SAM3_URL} and wait until the page says "
        "access has been granted.\n"
        f"2. Create a Read token at {HF_TOKENS_URL}.\n"
        "3. Paste the token in the Hugging Face field on the Model tab, click "
        "Log in, then load the model again."
    )


def add_mask_attributes(
    path: str,
    mask_scores: Optional[Mapping] = None,
    source: Optional[str] = None,
    prompt: Optional[str] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, object]:
    """Add ``score``, ``source`` and ``prompt`` fields to a saved mask layer.

    ``score`` is SAM's predicted confidence for the mask each polygon came
    from (its predicted IoU / detection score, as reported by SamGeo in
    ``mask_scores``). Polygons are matched to masks through the mask-id column
    written by the vectorizer (``class`` or ``value``). ``source`` is the
    segmented image's file name and ``prompt`` describes the prompt (for
    example ``text: tree``). Works for GeoJSON, GeoPackage and Shapefile
    outputs; field names stay within the 10-character shapefile limit.

    Must run on the main thread (it uses the QGIS vector data provider).
    Problems are reported through ``log`` and never raised.

    Args:
        path: Path to the vector file.
        mask_scores: Mapping of mask id to confidence, or None to skip
            ``score``.
        source: Source image file name, or None to skip ``source``.
        prompt: Prompt description, or None to skip ``prompt``.
        log: Optional callable that receives status messages.

    Returns:
        A summary dict with ``fields`` (names added or updated) and
        ``scored`` (number of features that received a score).
    """
    summary: Dict[str, object] = {"fields": [], "scored": 0}

    def _log(msg: str) -> None:
        if log is not None:
            log(msg)

    scores = normalize_mask_scores(mask_scores)
    if mask_scores is not None and not scores:
        _log("No mask confidence scores available; skipping the score field.")

    try:
        from qgis.core import (
            QgsFeatureRequest,
            QgsField,
            QgsVectorDataProvider,
            QgsVectorLayer,
        )
    except ImportError as exc:  # pragma: no cover - only outside QGIS
        _log(f"QGIS is not available to write attributes: {exc}")
        return summary

    layer = QgsVectorLayer(path, "samgeo_attributes", "ogr")
    provider = None
    try:
        if not layer.isValid():
            _log(f"Could not open {path} to add attributes.")
            return summary
        provider = layer.dataProvider()
        caps = provider.capabilities()
        if not (caps & QgsVectorDataProvider.Capability.AddAttributes) or not (
            caps & QgsVectorDataProvider.Capability.ChangeAttributeValues
        ):
            _log(f"{path} does not support adding attributes; skipping.")
            return summary

        is_shapefile = path.lower().endswith(".shp")
        existing = [f.name() for f in provider.fields()]
        id_field = find_mask_id_field(existing)

        wanted = []  # (name, kind, value_or_None)
        if scores:
            if id_field is None:
                _log(
                    "No mask id column (class/value) found in the output; "
                    "skipping the score field."
                )
            else:
                wanted.append((field_name("score", is_shapefile), "double", None))
        if source is not None:
            wanted.append(
                (
                    field_name("source", is_shapefile),
                    "string",
                    _clip(source, is_shapefile),
                )
            )
        if prompt is not None:
            wanted.append(
                (
                    field_name("prompt", is_shapefile),
                    "string",
                    _clip(prompt, is_shapefile),
                )
            )
        if not wanted:
            return summary

        # Read feature ids and mask ids before changing the schema: iterating a
        # GeoJSON layer after addAttributes makes GDAL report "Invalid index".
        id_idx = provider.fields().lookupField(id_field) if id_field else -1
        request = QgsFeatureRequest()
        request.setFlags(QgsFeatureRequest.Flag.NoGeometry)
        if id_idx >= 0:
            request.setSubsetOfAttributes([id_idx])
        else:
            request.setNoAttributes()
        features = [
            (f.id(), mask_id_from_value(f.attribute(id_idx)) if id_idx >= 0 else None)
            for f in provider.getFeatures(request)
        ]

        existing_lower = {n.lower() for n in existing}
        new_fields = [
            _make_field(QgsField, name, kind)
            for name, kind, _ in wanted
            if name.lower() not in existing_lower
        ]
        if new_fields:
            if not provider.addAttributes(new_fields):
                _log(f"Failed to add fields to {path}.")
                return summary
            layer.updateFields()

        fields = provider.fields()
        indices = {name: fields.lookupField(name) for name, _, _ in wanted}
        if any(idx < 0 for idx in indices.values()):
            _log(f"Could not find the new fields in {path}.")
            return summary
        changes = {}
        scored = 0
        for fid, mask_id in features:
            attrs = {}
            for name, kind, value in wanted:
                idx = indices[name]
                if kind == "double":
                    if mask_id is not None and mask_id in scores:
                        attrs[idx] = float(scores[mask_id])
                        scored += 1
                else:
                    attrs[idx] = value
            if attrs:
                changes[fid] = attrs

        if changes and not provider.changeAttributeValues(changes):
            _log(f"Failed to write attribute values to {path}.")
            return summary

        summary["fields"] = [name for name, _, _ in wanted]
        summary["scored"] = scored
        if scores and id_field:
            _log(f"Added confidence scores to {scored} feature(s).")
    finally:
        # Release the OGR handle on every path: on Windows a lingering
        # handle can lock the file while the caller loads it into QGIS.
        provider = None
        layer = None
    return summary


def _clip(text: str, is_shapefile: bool) -> str:
    """Trim text to the shapefile string limit (254 UTF-8 bytes) when needed.

    Args:
        text: The value to store.
        is_shapefile: Whether the output is an ESRI Shapefile.

    Returns:
        The possibly truncated text.
    """
    text = str(text)
    if not is_shapefile:
        return text
    # The DBF field is 254 bytes wide; cut by UTF-8 bytes and drop any
    # partial trailing character.
    encoded = text.encode("utf-8")[:SHAPEFILE_STRING_MAX]
    return encoded.decode("utf-8", errors="ignore")


def _make_field(field_cls, name: str, kind: str):
    """Create a ``QgsField`` across QGIS 3 (QVariant) and QGIS 3.38+/4 (QMetaType).

    Args:
        field_cls: The ``QgsField`` class.
        name: Field name.
        kind: ``"double"`` or ``"string"``.

    Returns:
        A new ``QgsField``.
    """
    length = SHAPEFILE_STRING_MAX if kind == "string" else 0
    try:
        from qgis.PyQt.QtCore import QMetaType

        qtype = QMetaType.Type.Double if kind == "double" else QMetaType.Type.QString
        return field_cls(name, qtype, len=length)
    except (ImportError, AttributeError, TypeError):
        from qgis.PyQt.QtCore import QVariant

        vtype = QVariant.Double if kind == "double" else QVariant.String
        return field_cls(name, vtype, len=length)
