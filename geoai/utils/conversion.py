"""Data conversion and coordinate transformation utilities."""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import leafmap
import numpy as np
import rasterio
import xarray as xr

__all__ = [
    "dict_to_rioxarray",
    "dict_to_image",
    "coords_to_xy",
    "rowcol_to_xy",
    "bbox_to_xy",
]


def dict_to_rioxarray(data_dict: Dict) -> xr.DataArray:
    """Convert a dictionary to a xarray DataArray. The dictionary should contain the
    following keys: "crs", "bounds", and "image". It can be generated from a TorchGeo
    dataset sampler.

    Args:
        data_dict (Dict): The dictionary containing the data.

    Returns:
        xr.DataArray: The xarray DataArray.
    """

    from collections import namedtuple

    from affine import Affine

    BoundingBox = namedtuple("BoundingBox", ["minx", "maxx", "miny", "maxy"])

    # Extract components from the dictionary
    crs = data_dict["crs"]
    bounds = data_dict["bounds"]
    image_tensor = data_dict["image"]

    if hasattr(bounds, "left"):
        bounds = BoundingBox(bounds.left, bounds.right, bounds.bottom, bounds.top)

    # Convert tensor to numpy array if needed
    if hasattr(image_tensor, "numpy"):
        # For PyTorch tensors
        image_array = image_tensor.numpy()
    else:
        # If it's already a numpy array or similar
        image_array = np.array(image_tensor)

    # Calculate pixel resolution
    width = image_array.shape[2]  # Width is the size of the last dimension
    height = image_array.shape[1]  # Height is the size of the middle dimension

    res_x = (bounds.maxx - bounds.minx) / width
    res_y = (bounds.maxy - bounds.miny) / height

    # Create the transform matrix
    transform = Affine(res_x, 0.0, bounds.minx, 0.0, -res_y, bounds.maxy)

    # Create dimensions
    x_coords = np.linspace(bounds.minx + res_x / 2, bounds.maxx - res_x / 2, width)
    y_coords = np.linspace(bounds.maxy - res_y / 2, bounds.miny + res_y / 2, height)

    # If time dimension exists in the bounds
    if hasattr(bounds, "mint") and hasattr(bounds, "maxt"):
        # Create a single time value or range if needed
        t_coords = [
            bounds.mint
        ]  # Or np.linspace(bounds.mint, bounds.maxt, num_time_steps)

        # Create DataArray with time dimension
        dims = (
            ("band", "y", "x")
            if image_array.shape[0] <= 10
            else ("time", "band", "y", "x")
        )

        if dims[0] == "band":
            # For multi-band single time
            da = xr.DataArray(
                image_array,
                dims=dims,
                coords={
                    "band": np.arange(1, image_array.shape[0] + 1),
                    "y": y_coords,
                    "x": x_coords,
                },
            )
        else:
            # For multi-time multi-band
            da = xr.DataArray(
                image_array,
                dims=dims,
                coords={
                    "time": t_coords,
                    "band": np.arange(1, image_array.shape[1] + 1),
                    "y": y_coords,
                    "x": x_coords,
                },
            )
    else:
        # Create DataArray without time dimension
        da = xr.DataArray(
            image_array,
            dims=("band", "y", "x"),
            coords={
                "band": np.arange(1, image_array.shape[0] + 1),
                "y": y_coords,
                "x": x_coords,
            },
        )

    # Set spatial attributes
    da.rio.write_crs(crs, inplace=True)
    da.rio.write_transform(transform, inplace=True)

    return da


def dict_to_image(
    data_dict: Dict[str, Any], output: Optional[str] = None, **kwargs: Any
) -> Union[str, Any]:
    """Convert a dictionary containing spatial data to a rasterio dataset or save it to
    a file. The dictionary should contain the following keys: "crs", "bounds", and "image".
    It can be generated from a TorchGeo dataset sampler.

    This function transforms a dictionary with CRS, bounding box, and image data
    into a rasterio DatasetReader using leafmap's array_to_image utility after
    first converting to a rioxarray DataArray.

    Args:
        data_dict: A dictionary containing:
            - 'crs': A pyproj CRS object
            - 'bounds': A BoundingBox object with minx, maxx, miny, maxy attributes
              and optionally mint, maxt for temporal bounds
            - 'image': A tensor or array-like object with image data
        output: Optional path to save the image to a file. If not provided, the image
            will be returned as a rasterio DatasetReader object.
        **kwargs: Additional keyword arguments to pass to leafmap.array_to_image.
            Common options include:
            - colormap: str, name of the colormap (e.g., 'viridis', 'terrain')
            - vmin: float, minimum value for colormap scaling
            - vmax: float, maximum value for colormap scaling

    Returns:
        A rasterio DatasetReader object that can be used for visualization or
        further processing.

    Examples:
        >>> image = dict_to_image(
        ...     {'crs': CRS.from_epsg(26911), 'bounds': bbox, 'image': tensor},
        ...     colormap='terrain'
        ... )
        >>> fig, ax = plt.subplots(figsize=(10, 10))
        >>> show(image, ax=ax)
    """
    da = dict_to_rioxarray(data_dict)

    if output is not None:
        out_dir = os.path.abspath(os.path.dirname(output))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        da.rio.to_raster(output)
        return output
    else:
        image = leafmap.array_to_image(da, **kwargs)
        return image


def coords_to_xy(
    src_fp: str,
    coords: np.ndarray,
    coord_crs: str = "epsg:4326",
    return_out_of_bounds: bool = False,
    **kwargs: Any,
) -> np.ndarray:
    """Converts a list or array of coordinates to pixel coordinates, i.e., (col, row) coordinates.

    Args:
        src_fp: The source raster file path.
        coords: A 2D or 3D array of coordinates. Can be of shape [[x1, y1], [x2, y2], ...]
                or [[[x1, y1]], [[x2, y2]], ...].
        coord_crs: The coordinate CRS of the input coordinates. Defaults to "epsg:4326".
        return_out_of_bounds: Whether to return out-of-bounds coordinates. Defaults to False.
        **kwargs: Additional keyword arguments to pass to rasterio.transform.rowcol.

    Returns:
        A 2D or 3D array of pixel coordinates in the same format as the input.
    """
    from rasterio.warp import transform as transform_coords

    out_of_bounds = []
    if isinstance(coords, np.ndarray):
        input_is_3d = coords.ndim == 3  # Check if the input is a 3D array
    else:
        input_is_3d = False

    # Flatten the 3D array to 2D if necessary
    if input_is_3d:
        original_shape = coords.shape  # Store the original shape
        coords = coords.reshape(-1, 2)  # Flatten to 2D

    # Convert ndarray to a list if necessary
    if isinstance(coords, np.ndarray):
        coords = coords.tolist()

    xs, ys = zip(*coords)
    with rasterio.open(src_fp) as src:
        width = src.width
        height = src.height
        if coord_crs != src.crs:
            xs, ys = transform_coords(coord_crs, src.crs, xs, ys, **kwargs)
        rows, cols = rasterio.transform.rowcol(src.transform, xs, ys, **kwargs)

    result = [[col, row] for col, row in zip(cols, rows)]

    output = []

    for i, (x, y) in enumerate(result):
        if x >= 0 and y >= 0 and x < width and y < height:
            output.append([x, y])
        else:
            out_of_bounds.append(i)

    # Convert the output back to the original shape if input was 3D
    output = np.array(output)
    if input_is_3d:
        output = output.reshape(original_shape)

    # Handle cases where no valid pixel coordinates are found
    if len(output) == 0:
        print("No valid pixel coordinates found.")
    elif len(output) < len(coords):
        print("Some coordinates are out of the image boundary.")

    if return_out_of_bounds:
        return output, out_of_bounds
    else:
        return output


def rowcol_to_xy(
    src_fp: str,
    rows: Optional[List[int]] = None,
    cols: Optional[List[int]] = None,
    boxes: Optional[List[List[int]]] = None,
    zs: Optional[List[float]] = None,
    offset: str = "center",
    output: Optional[str] = None,
    dst_crs: str = "EPSG:4326",
    **kwargs: Any,
) -> Tuple[List[float], List[float]]:
    """Converts a list of (row, col) coordinates to (x, y) coordinates.

    Args:
        src_fp (str): The source raster file path.
        rows (list, optional): A list of row coordinates. Defaults to None.
        cols (list, optional): A list of col coordinates. Defaults to None.
        boxes (list, optional): A list of (row, col) coordinates in the format of [[left, top, right, bottom], [left, top, right, bottom], ...]
        zs: zs (list or float, optional): Height associated with coordinates. Primarily used for RPC based coordinate transformations.
        offset (str, optional): Determines if the returned coordinates are for the center of the pixel or for a corner.
        output (str, optional): The output vector file path. Defaults to None.
        dst_crs (str, optional): The destination CRS. Defaults to "EPSG:4326".
        **kwargs: Additional keyword arguments to pass to rasterio.transform.xy.

    Returns:
        A list of (x, y) coordinates.
    """

    if boxes is not None:
        rows = []
        cols = []

        for box in boxes:
            rows.append(box[1])
            rows.append(box[3])
            cols.append(box[0])
            cols.append(box[2])

    if rows is None or cols is None:
        raise ValueError("rows and cols must be provided.")

    with rasterio.open(src_fp) as src:
        xs, ys = rasterio.transform.xy(src.transform, rows, cols, zs, offset, **kwargs)
        src_crs = src.crs

    if boxes is None:
        return [[x, y] for x, y in zip(xs, ys)]
    else:
        result = [[xs[i], ys[i + 1], xs[i + 1], ys[i]] for i in range(0, len(xs), 2)]

        if output is not None:
            from .vector import boxes_to_vector

            boxes_to_vector(result, src_crs, dst_crs, output)
        else:
            return result


def bbox_to_xy(
    src_fp: str, coords: List[float], coord_crs: str = "epsg:4326", **kwargs: Any
) -> List[float]:
    """Converts a list of coordinates to pixel coordinates, i.e., (col, row) coordinates.
        Note that map bbox coords is [minx, miny, maxx, maxy] from bottomleft to topright
        While rasterio bbox coords is [minx, max, maxx, min] from topleft to bottomright

    Args:
        src_fp (str): The source raster file path.
        coords (list): A list of coordinates in the format of [[minx, miny, maxx, maxy], [minx, miny, maxx, maxy], ...]
        coord_crs (str, optional): The coordinate CRS of the input coordinates. Defaults to "epsg:4326".

    Returns:
        list: A list of pixel coordinates in the format of [[minx, maxy, maxx, miny], ...] from top left to bottom right.
    """

    if isinstance(coords, str):
        gdf = gpd.read_file(coords)
        coords = gdf.geometry.bounds.values.tolist()
        if gdf.crs is not None:
            coord_crs = f"epsg:{gdf.crs.to_epsg()}"
    elif isinstance(coords, np.ndarray):
        coords = coords.tolist()
    if isinstance(coords, dict):
        import json

        geojson = json.dumps(coords)
        gdf = gpd.read_file(geojson, driver="GeoJSON")
        coords = gdf.geometry.bounds.values.tolist()

    elif not isinstance(coords, list):
        raise ValueError("coords must be a list of coordinates.")

    if not isinstance(coords[0], list):
        coords = [coords]

    new_coords = []

    with rasterio.open(src_fp) as src:
        width = src.width
        height = src.height

        for coord in coords:
            minx, miny, maxx, maxy = coord

            if coord_crs != src.crs:
                from rasterio.warp import transform as transform_coords

                minx, miny = transform_coords(minx, miny, coord_crs, src.crs, **kwargs)
                maxx, maxy = transform_coords(maxx, maxy, coord_crs, src.crs, **kwargs)

                rows1, cols1 = rasterio.transform.rowcol(
                    src.transform, minx, miny, **kwargs
                )
                rows2, cols2 = rasterio.transform.rowcol(
                    src.transform, maxx, maxy, **kwargs
                )

                new_coords.append([cols1, rows1, cols2, rows2])

            else:
                new_coords.append([minx, miny, maxx, maxy])

    result = []

    for coord in new_coords:
        minx, miny, maxx, maxy = coord

        if (
            minx >= 0
            and miny >= 0
            and maxx >= 0
            and maxy >= 0
            and minx < width
            and miny < height
            and maxx < width
            and maxy < height
        ):
            # Note that map bbox coords is [minx, miny, maxx, maxy] from bottomleft to topright
            # While rasterio bbox coords is [minx, max, maxx, min] from topleft to bottomright
            result.append([minx, maxy, maxx, miny])

    if len(result) == 0:
        print("No valid pixel coordinates found.")
        return None
    elif len(result) == 1:
        return result[0]
    elif len(result) < len(coords):
        print("Some coordinates are out of the image boundary.")

    return result
