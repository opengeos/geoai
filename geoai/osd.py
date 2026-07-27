import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Optional, Union

_osd_lock = threading.Lock()


def classify_optically_shallow_deep(
    image: Union[str, Path],
    output: Union[str, Path],
    acolite_l2r: Optional[Union[str, Path]] = None,
    to_log: bool = False,
    overwrite: bool = True,
) -> str:
    """Classify Sentinel-2 imagery into optically shallow and deep water.

    Args:
        image: Path to the input Sentinel-2 L1C (.SAFE) directory.
        output: Path to save the output probability GeoTIFF.
        acolite_l2r: Optional path to the ACOLITE L2R NetCDF (.nc) file.
        to_log: Whether to log the pre-processing and prediction steps.
        overwrite: Whether to overwrite the existing output file.

    Returns:
        Path to the output probability GeoTIFF as a string.
    """
    try:
        from opticallyshallowdeep.run import run as osd_run
    except ImportError:
        raise ImportError(
            "The 'opticallyshallowdeep' package is required for classify_optically_shallow_deep. "
            "Please install it using 'pip install opticallyshallowdeep'."
        )

    image_path = Path(image)
    out_path = Path(output)
    l2r_path = Path(acolite_l2r) if acolite_l2r else None

    if not overwrite and out_path.exists():
        raise FileExistsError(f"Output file {out_path} already exists.")

    import tifffile
    import tensorflow as tf
    import keras
    import unittest.mock

    # Patch Keras activation functions to support legacy camelCase LeakyReLU name in Keras 3
    try:
        import keras.src.activations

        keras_modules = [keras, tf.keras, keras.src.activations]
    except ImportError:
        keras_modules = [keras, tf.keras]

    with _osd_lock:
        original_getters = []
        patched_targets = set()
        for m in keras_modules:
            if m is None:
                continue
            if hasattr(m, "get") and (id(m), "get") not in patched_targets:
                orig_get = m.get

                def patched_get(identifier, orig=orig_get):
                    if identifier == "LeakyReLU":
                        return orig("leaky_relu")
                    return orig(identifier)

                original_getters.append((m, "get", orig_get))
                patched_targets.add((id(m), "get"))
                m.get = patched_get
            if (
                hasattr(m, "activations")
                and hasattr(m.activations, "get")
                and (id(m.activations), "get") not in patched_targets
            ):
                orig_get = m.activations.get

                def patched_get(identifier, orig=orig_get):
                    if identifier == "LeakyReLU":
                        return orig("leaky_relu")
                    return orig(identifier)

                original_getters.append((m.activations, "get", orig_get))
                patched_targets.add((id(m.activations), "get"))
                m.activations.get = patched_get

        original_imread = tifffile.imread

        def patched_imread(*args, **kwargs):
            dtype = kwargs.pop("dtype", None)
            img = original_imread(*args, **kwargs)
            if dtype is not None:
                return img.astype(dtype)
            return img

        tf_device_patch = unittest.mock.patch(
            "tensorflow.device", return_value=unittest.mock.MagicMock()
        )

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                with (
                    unittest.mock.patch("tifffile.imread", side_effect=patched_imread),
                    tf_device_patch,
                ):
                    osd_run(
                        file_L1C=str(image_path),
                        folder_out=temp_dir,
                        file_L2R=str(l2r_path) if l2r_path else None,
                        to_log=to_log,
                    )

                generated = list(Path(temp_dir).glob("*_OSW_ODW.tif"))
                if not generated:
                    raise RuntimeError(
                        "opticallyshallowdeep failed to generate output."
                    )

                out_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(generated[0], out_path)
        finally:
            for target_obj, attr_name, orig_func in original_getters:
                setattr(target_obj, attr_name, orig_func)

    return str(out_path)
