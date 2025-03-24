"""Main module."""

import leafmap

from .download import (
    download_naip,
    download_overture_buildings,
    download_pc_stac_item,
    pc_collection_list,
    pc_item_asset_list,
    pc_stac_search,
    pc_stac_download,
    read_pc_item_asset,
    view_pc_item,
)
from .classify import train_classifier, classify_image, classify_images
from .extract import *
from .hf import *
from .segment import *
from .train import object_detection, train_MaskRCNN_model
from .utils import *


class Map(leafmap.Map):
    """A subclass of leafmap.Map for GeoAI applications."""

    def __init__(self, *args, **kwargs):
        """Initialize the Map class."""
        super().__init__(*args, **kwargs)
