import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "geoai"))

import math
import geopandas as gpd
import numpy as np
import pandas as pd
import pystac_client
import stackstac
import torch
import yaml
from box import Box
from rasterio.enums import Resampling
from shapely import Point
from torchvision.transforms import v2
import datetime

from claymodel.module import ClayMAEModule
from geoai.clay import Clay


def get_data():
    lat, lon = 37.30939, -8.57207
    start, end = "2018-07-01", "2018-09-01"
    STAC_API = "https://earth-search.aws.element84.com/v1"
    COLLECTION = "sentinel-2-l2a"

    catalog = pystac_client.Client.open(STAC_API)
    search = catalog.search(
        collections=[COLLECTION],
        datetime=f"{start}/{end}",
        bbox=(lon - 1e-5, lat - 1e-5, lon + 1e-5, lat + 1e-5),
        max_items=100,
        query={"eo:cloud_cover": {"lt": 80}},
    )

    all_items = search.get_all_items()
    items = []
    dates = []
    for item in all_items:
        if item.datetime.date() not in dates:
            items.append(item)
            dates.append(item.datetime.date())

    epsg_str = items[0].properties["proj:code"]
    epsg = int(epsg_str.split(":")[-1])

    poidf = gpd.GeoDataFrame(
        pd.DataFrame(), crs="EPSG:4326", geometry=[Point(lon, lat)]
    ).to_crs(epsg_str)
    coords = poidf.iloc[0].geometry.coords[0]

    size, gsd = 256, 10
    bounds = (
        coords[0] - (size * gsd) // 2,
        coords[1] - (size * gsd) // 2,
        coords[0] + (size * gsd) // 2,
        coords[1] + (size * gsd) // 2,
    )

    stack = stackstac.stack(
        items,
        bounds=bounds,
        snap_bounds=False,
        epsg=epsg,
        resolution=gsd,
        dtype="float64",
        rescale=False,
        fill_value=np.nan,
        assets=["blue", "green", "red", "nir"],
        resampling=Resampling.nearest,
    ).compute()

    return stack, lat, lon, bounds, epsg_str, gsd


def generate_embeddings_core_clay(stack, lat, lon):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    model = (
        ClayMAEModule.load_from_checkpoint(
            "~/.cache/clay/clay-v1.5.ckpt",
            model_size="large",
            metadata_path="geoai/config/clay_metadata.yaml",
            dolls=[16, 32, 64, 128, 256, 768, 1024],
            doll_weights=[1, 1, 1, 1, 1, 1, 1],
            mask_ratio=0.0,
            shuffle=False,
        )
        .eval()
        .to(device)
    )

    platform = "sentinel-2-l2a"
    metadata = Box(yaml.safe_load(open("geoai/config/clay_metadata.yaml")))

    mean = [metadata[platform].bands.mean[str(band.values)] for band in stack.band]
    std = [metadata[platform].bands.std[str(band.values)] for band in stack.band]
    waves = [
        metadata[platform].bands.wavelength[str(band.values)] for band in stack.band
    ]

    transform = v2.Compose([v2.Normalize(mean=mean, std=std)])

    def normalize_timestamp(date):
        week = date.isocalendar().week * 2 * np.pi / 52
        hour = date.hour * 2 * np.pi / 24
        return (math.sin(week), math.cos(week)), (math.sin(hour), math.cos(hour))

    def normalize_latlon(lat, lon):
        lat, lon = lat * np.pi / 180, lon * np.pi / 180
        return (math.sin(lat), math.cos(lat)), (math.sin(lon), math.cos(lon))

    datetimes = stack.time.values.astype("datetime64[s]").tolist()
    times = [normalize_timestamp(dt) for dt in datetimes]
    week_norm = [t[0] for t in times]
    hour_norm = [t[1] for t in times]

    latlons = [normalize_latlon(lat, lon)] * len(times)
    lat_norm = [ll[0] for ll in latlons]
    lon_norm = [ll[1] for ll in latlons]

    pixels = transform(torch.from_numpy(stack.data.astype(np.float32)))

    datacube = {
        "platform": platform,
        "time": torch.tensor(
            np.hstack((week_norm, hour_norm)), dtype=torch.float32, device=device
        ),
        "latlon": torch.tensor(
            np.hstack((lat_norm, lon_norm)), dtype=torch.float32, device=device
        ),
        "pixels": pixels.to(device),
        "gsd": torch.tensor(stack.gsd.values, device=device),
        "waves": torch.tensor(waves, device=device),
    }

    with torch.no_grad():
        unmsk_patch, _, _, _ = model.model.encoder(datacube)
        # Return ALL embeddings (CLS + all patches)
        all_embeddings = unmsk_patch.cpu().numpy()
        # Also return just CLS tokens for comparison
        cls_embeddings = unmsk_patch[:, 0, :].cpu().numpy()

    return all_embeddings, cls_embeddings


def generate_embeddings_wrapper(stack, lat, lon, bounds, epsg_str, gsd):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    custom_metadata = {
        "band_order": ["blue", "green", "red", "nir"],
        "rgb_indices": [2, 1, 0],
        "gsd": 10,
        "bands": {
            "mean": {"blue": 1105.0, "green": 1355.0, "red": 1552.0, "nir": 2743.0},
            "std": {"blue": 1809.0, "green": 1757.0, "red": 1888.0, "nir": 1742.0},
            "wavelength": {"blue": 0.493, "green": 0.56, "red": 0.665, "nir": 0.842},
        },
    }

    clay_model = Clay(custom_metadata=custom_metadata, device=str(device))

    bounds_gdf = gpd.GeoDataFrame(
        geometry=[Point(bounds[0], bounds[1]), Point(bounds[2], bounds[3])],
        crs=epsg_str,
    ).to_crs("EPSG:4326")

    wgs84_bounds = (
        bounds_gdf.iloc[0].geometry.x,
        bounds_gdf.iloc[0].geometry.y,
        bounds_gdf.iloc[1].geometry.x,
        bounds_gdf.iloc[1].geometry.y,
    )

    all_embeddings_list = []
    cls_embeddings_list = []
    datetimes = stack.time.values.astype("datetime64[s]").tolist()

    for i, datetime_obj in enumerate(datetimes):
        image = stack[i].values.transpose(1, 2, 0)

        if hasattr(datetime_obj, "astype"):
            timestamp = datetime_obj.astype("datetime64[s]").astype("int")
            date = datetime.datetime.fromtimestamp(timestamp)
        else:
            date = datetime_obj

        # Get ALL embeddings
        all_embedding = clay_model.generate(
            image=image, bounds=wgs84_bounds, date=date, gsd=gsd, only_cls_token=False
        )
        all_embeddings_list.append(all_embedding.squeeze(0).cpu().numpy())

        # Get CLS token only
        cls_embedding = clay_model.generate(
            image=image, bounds=wgs84_bounds, date=date, gsd=gsd, only_cls_token=True
        )
        cls_embeddings_list.append(cls_embedding.cpu().numpy())

    return np.stack(all_embeddings_list), np.vstack(cls_embeddings_list)


def main():
    print("Simple allclose test - ALL tokens vs CLS tokens")

    print("Loading data...")
    stack, lat, lon, bounds, epsg_str, gsd = get_data()

    print("Generating embeddings using core Clay...")
    all_core, cls_core = generate_embeddings_core_clay(stack, lat, lon)

    print("Generating embeddings using wrapper...")
    all_wrapper, cls_wrapper = generate_embeddings_wrapper(
        stack, lat, lon, bounds, epsg_str, gsd
    )

    print(f"\nShapes:")
    print(f"Core all embeddings: {all_core.shape}")
    print(f"Wrapper all embeddings: {all_wrapper.shape}")
    print(f"Core CLS embeddings: {cls_core.shape}")
    print(f"Wrapper CLS embeddings: {cls_wrapper.shape}")

    print(f"\n=== CLS TOKENS ONLY ===")
    print("Numpy allclose:")
    print(f"  1e-3: {np.allclose(cls_core, cls_wrapper, atol=1e-3)}")
    print(f"  1e-4: {np.allclose(cls_core, cls_wrapper, atol=1e-4)}")
    print(f"  1e-5: {np.allclose(cls_core, cls_wrapper, atol=1e-5)}")

    cls_core_tensor = torch.from_numpy(cls_core)
    cls_wrapper_tensor = torch.from_numpy(cls_wrapper)

    print("Torch allclose:")
    print(f"  1e-3: {torch.allclose(cls_core_tensor, cls_wrapper_tensor, atol=1e-3)}")
    print(f"  1e-4: {torch.allclose(cls_core_tensor, cls_wrapper_tensor, atol=1e-4)}")
    print(f"  1e-5: {torch.allclose(cls_core_tensor, cls_wrapper_tensor, atol=1e-5)}")

    print(f"\n=== ALL TOKENS (CLS + ALL PATCHES) ===")
    print("Numpy allclose:")
    print(f"  1e-3: {np.allclose(all_core, all_wrapper, atol=1e-3)}")
    print(f"  1e-4: {np.allclose(all_core, all_wrapper, atol=1e-4)}")
    print(f"  1e-5: {np.allclose(all_core, all_wrapper, atol=1e-5)}")

    all_core_tensor = torch.from_numpy(all_core)
    all_wrapper_tensor = torch.from_numpy(all_wrapper)

    print("Torch allclose:")
    print(f"  1e-3: {torch.allclose(all_core_tensor, all_wrapper_tensor, atol=1e-3)}")
    print(f"  1e-4: {torch.allclose(all_core_tensor, all_wrapper_tensor, atol=1e-4)}")
    print(f"  1e-5: {torch.allclose(all_core_tensor, all_wrapper_tensor, atol=1e-5)}")


if __name__ == "__main__":
    main()
