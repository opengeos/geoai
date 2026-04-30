"""ESRGAN - Enhanced Super-Resolution Generative Adversarial Networks"""

from glob import glob
from typing import List, LiteralString, Tuple
import math
import os
import sys

from affine import Affine
from osgeo import gdal
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import reproject
from skimage.exposure import match_histograms
from torch.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, random_split, SubsetRandomSampler
from tqdm import tqdm
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.models as models

from .utils.device import get_device

# gdal related exceptions are built into processing functionality
gdal.DontUseExceptions()


class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block for ESRGAN.

    This block consists of multiple convolutional layers with dense connections,
    allowing the network to learn more complex features and improve image quality.
    """

    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(nf + gc * 2, gc, 3, 1, 1, bias=False)
        self.conv4 = nn.Conv2d(nf + gc * 3, gc, 3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(nf + gc * 4, nf, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """
    Residual-in-Residual Dense Block for ESRGAN.

    This block consists of multiple ResidualDenseBlocks stacked together,
    allowing the network to learn more complex features and improve image quality.
    """

    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class ESRGANGenerator(nn.Module):
    """
    ESRGAN Generator.

    Args:
        in_nc (int): Number of input channels.
        out_nc (int): Number of output channels.
        nf (int): Number of filters.
        nb (int): Number of blocks.
        gc (int): Growth channel.
        scale (int): Scale factor.
    """

    def __init__(self, in_nc=1, out_nc=1, nf=64, nb=16, gc=32, scale=4):
        super().__init__()
        self.scale = scale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=False)
        self.rrdb_trunk = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=False)

        upsampler = []
        stages = int(torch.log2(torch.tensor(scale)).item())
        for _ in range(stages):
            upsampler.append(nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=False))
            upsampler.append(nn.PixelShuffle(2))
            upsampler.append(nn.LeakyReLU(0.2, inplace=True))
        self.upsampler = nn.Sequential(*upsampler)

        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=False)
        self.conv_flat = nn.Conv2d(1, 1, 3, 1, 1)
        self.activation_fin = nn.Tanh()
        self.apply(self._weights_init)

    def _weights_init(self, model: nn.Module):
        if isinstance(model, nn.Conv2d):
            nn.init.kaiming_normal_(model.weight, a=0.2)
            if model.bias is not None:
                nn.init.zeros_(model.bias)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ESRGAN generator.

        Args:
            input_tensor (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Approach: refine features with an RRDB trunk, upsample via pixel-shuffle, then add a bicubic-interpolated skip (residual) base.
        b, c, h, w = input_tensor.shape
        hr_h, hr_w = h * self.scale, w * self.scale
        base = F.interpolate(
            input_tensor, size=(hr_h, hr_w), mode="bicubic", align_corners=False
        )

        processed_input = self.conv_first(input_tensor)
        trunk = self.trunk_conv(self.rrdb_trunk(processed_input))
        fea = processed_input + trunk

        output_tensor = self.upsampler(fea)
        output_tensor = self.conv_last(output_tensor)
        output_tensor = output_tensor + base

        return output_tensor


class Discriminator(nn.Module):
    """
    Discriminator for ESRGAN.
    """

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1, 1),
        )
        self.apply(self.weights_init)

    def weights_init(self, model: nn.Module):
        if isinstance(model, nn.Conv2d):
            nn.init.kaiming_normal_(
                model.weight, mode="fan_in", nonlinearity="leaky_relu"
            )
            if model.bias is not None:
                model.bias.data.fill_(0)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.model(input_tensor).view(-1, 1)


class VGGPerceptualLoss(nn.Module):
    """
    VGG perceptual loss for ESRGAN.

    Uses a pre-trained VGG19 model to extract features and computes
    perceptual loss based on the difference in feature space.

    This loss helps generate images that are not only visually similar
    but also perceptually similar to the target images.
    """

    def __init__(self, layer_index=9):
        """
        layer_index: index in features list to cut at
                     (e.g., 9 ≈ relu2_2 for VGG19)
        """
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19().features
        self.slice = nn.Sequential(*[vgg[i] for i in range(layer_index + 1)])
        for p in self.slice.parameters():
            p.requires_grad = False  # freeze VGG params
        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        # Assume x and y are _normalized to ImageNet stats
        x_vgg = self.slice(x)
        y_vgg = self.slice(y)
        return self.criterion(x_vgg, y_vgg)


class NormalizeToVGG(nn.Module):
    """
    Normalizes tensors to the range expected by VGG19 (ImageNet statistics).
    """

    def __init__(self):
        super(NormalizeToVGG, self).__init__()
        # ImageNet statistics
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # actual per-image tensor normalization using ImageNet statistics
        return (input_tensor - self.mean) / self.std


class ESRGANDataPreprocess:
    """
    Data preprocessing for ESRGAN training.

    Class handles decimation and tiling, normalization, and
    conversion to Tensor arrays.

    Target datasets should be higher-resolution images representing the goal
    or target resolution.  These data should have the same number of bands of
    your input datasets, and should have spatial resolution roughly
    equal to the input datasets resolution divided by the
    desired scaling factor.

    For example, if you want to scale 30m Landsat satellite Imagery (input dataset) to
    4 meters, inputs should have 4 meter resolution (for target dataset),
    and the band numbers should match in terms of spectral resolution/wavelength.

    Target and input datasets come in so many forms and potentially large number of
    files, so either provide a dictionary of band indices to file paths,
    or a single multi-band raster or virtual raster table (vrt).

    Outputs are tensors saved as zipped (.pt) files.

    Batch size will be used in processing; 8 (default) is a good starting point.

    Inputs:
        output_dir: str - Directory to save output files
        target_band_files: dict[int, str] - Dictionary of band indices to file paths for target data
        input_band_files: dict[int, str] - Dictionary of band indices to file paths for input data
        target_file: str | None - Path to target data file
        input_file: str | None - Path to input data file
        scale_factor: int - Scale factor for upscaling
        use_downsampled_targets: bool - Whether to use downsampled targets
        tile_size: int - Size of tiles to process
        overlap: int - Overlap between tiles
        batch_size: int - Batch size for processing

    Examples:
        >>> from geoai.esrgan import ESRGANDataPreprocess
        >>> preprocess = ESRGANDataPreprocess(
        ...     output_dir='./data',
        ...     target_band_files={1: 'target_band1.tif'},
        ...     input_band_files={1: 'input_band1.tif'},
        ...     scale_factor=4,
        ...     tile_size=128,
        ...     overlap=96,
        ...     batch_size=8,
        ... )
        >>> preprocess.preprocess()
    """

    def __init__(
        self,
        output_dir: str,
        target_band_files: dict[int, str] = {},
        input_band_files: dict[int, str] = {},
        target_file: str | None = None,
        input_file: str | None = None,
        scale_factor: int = None,
        use_downsampled_targets: bool = False,
        tile_size: int = 128,
        overlap: int = 96,
        batch_size: int = 8,
    ):
        assert (
            target_band_files is not None or target_file is not None
        ), "Either target_band_files or target_file must be provided"
        assert (
            input_band_files is not None or input_file is not None
        ) or use_downsampled_targets, (
            "Either input_band_files or input_file must be provided"
        )
        self.target_band_files = target_band_files
        self.input_band_files = input_band_files
        self.target_file = target_file
        self.input_file = input_file
        self.scale_factor = scale_factor
        self.use_downsampled_targets = use_downsampled_targets
        self.output_dir = output_dir
        self.tile_size = tile_size
        self.overlap = overlap

    def preprocess_single_inputs(
        self,
        ds: gdal.Dataset,
        band: int,
        resample_method: LiteralString[
            "bilinear",
            "cubic",
            "average",
            "cubicspline",
            "max",
            "min",
            "med",
            "nearest",
        ] = "bilinear",
    ) -> None:
        """
        Preprocess a single target band from a GDAL dataset.

        Intended to be used when training off of a single image dataset
        and using downsampled tiles as "low-res" inputs.  Increases
        accuracy but may decrease performance when applying
        inference on real-world low res images.

        Notably applies a scale_factor (class variable) to the input band.

        Args:
            ds: GDAL dataset
            band: Band number
            resample_method: Resampling method to use.
                             Must be one of: 'bilinear', 'cubic', 'average', 'cubicspline', 'max', 'min', 'med', 'nearest'
        """
        match (resample_method):
            case "bilinear":
                resample = gdal.GRA_Bilinear
            case "cubic":
                resample = gdal.GRA_Cubic
            case "average":
                resample = gdal.GRA_Average
            case "cubicspline":
                resample = gdal.GRA_CubicSpline
            case "max":
                resample = gdal.GRA_Max
            case "min":
                resample = gdal.GRA_Min
            case "med":
                resample = gdal.GRA_Med
            case "nearest":
                resample = gdal.GRA_NearestNeighbour
            case _:
                raise ValueError(f"Invalid resample method: {resample_method}")
        xmin, xres, xskew, ymax, yskew, yres = ds.GetGeoTransform()
        proj = ds.GetProjection()
        if band > 1 and ds.RasterCount >= band:
            bnd = ds.GetRasterBand(band)
        else:
            bnd = ds.GetRasterBand(1)
        nodata = bnd.GetNoDataValue()
        xcells = ds.RasterXSize
        ycells = ds.RasterYSize
        tile_size = self.tile_size
        overlap = self.overlap
        this_target_dir = os.path.join(self.output_dir, "target", str(band))
        this_input_dir = os.path.join(self.output_dir, "input", str(band))
        if not os.path.exists(this_target_dir):
            os.makedirs(this_target_dir, exist_ok=True)
        if not os.path.exists(this_input_dir):
            os.makedirs(this_input_dir, exist_ok=True)
        no_tiles_width = int(math.ceil(xcells / (tile_size - overlap)))
        no_tiles_height = int(math.ceil(ycells / (tile_size - overlap)))
        driver = gdal.GetDriverByName("GTiff")
        for i in range(no_tiles_width):
            for j in range(no_tiles_height):
                # Get offset and window size for full-res output tile at this location
                xoff = i * (tile_size - overlap)
                yoff = j * (tile_size - overlap)
                this_x_size = tile_size if xcells - xoff >= tile_size else xcells - xoff
                this_y_size = tile_size if ycells - yoff >= tile_size else ycells - yoff
                arr = bnd.ReadAsArray(xoff, yoff, this_x_size, this_y_size)
                out_ds = driver.Create(
                    os.path.join(this_target_dir, f"tile_band{band}_{i}_{j}.tif"),
                    this_x_size,
                    this_y_size,
                    1,
                    gdal.GDT_Float32,
                )
                out_ds.SetGeoTransform(
                    (xmin + xoff * xres, xres, xskew, ymax + yoff * yres, yskew, yres)
                )
                out_ds.SetProjection(proj)
                out_ds.GetRasterBand(1).WriteArray(arr)
                # Proceed to downsampled (input tile) version at same location
                this_xmin = xmin + xoff * xres
                this_ymax = ymax + yoff * yres
                this_xmax = this_xmin + this_x_size * xres
                this_ymin = this_ymax + this_y_size * yres
                warp_options = gdal.WarpOptions(
                    xRes=xres * self.scale_factor,
                    yRes=yres * self.scale_factor,
                    resampleAlg=resample,
                    outputBounds=[this_xmin, this_ymin, this_xmax, this_ymax],
                )
                input_ds = gdal.Warp(
                    os.path.join(this_input_dir, f"tile_band{band}_{i}_{j}.tif"),
                    out_ds,
                    options=warp_options,
                )
                input_ds = None
                out_ds = None

    def preprocess_band(
        self,
        ds: gdal.Dataset,
        band: int,
        is_target: bool = True,
        tile_ranges: List[Tuple[float, float, int, int, float, float]] = [],
    ) -> List[Tuple[float, float, int, int, float, float]]:
        """
        Preprocess a single band from a GDAL dataset.

        Read from input files with gdal using offsets:
        np.array = gdal.Band.ReadAsArray(xoff, yoff, xsize, ysize)

        "Target" band refers to training/destination resolution image

        Args:
            ds: GDAL dataset
            band: Band number
            is_target: Whether this is a target band
            tile_ranges: List of tile ranges

        Returns:
            List of tile ranges
        """
        sr = ds.GetSpatialRef().GetAttrValue("AUTHORITY", 1)
        if not is_target:
            if sr != tile_ranges[0][6]:
                # The inputs have a different spatial projection.  Reprojected intermediate file needed
                fname = ds.GetFileList()[0]
                reproj = os.path.join(
                    os.path.dirname(fname), f"reproj_{os.path.basename(fname)}"
                )
                ds = gdal.Warp(reproj, ds, dstSRS=f"EPSG:{tile_ranges[0][6]}")
                bnd = ds.GetRasterBand(1)
                proj = ds.GetProjection()
        xmin, xres, xskew, ymax, yskew, yres = ds.GetGeoTransform()
        if not is_target:
            tile_size = int(math.ceil(self.tile_size / (xres / tile_ranges[0][4])))
            overlap = int(math.ceil(self.overlap / (xres / tile_ranges[0][4])))
        else:
            tile_size = self.tile_size
            overlap = self.overlap
        this_output_dir = self.output_dir
        if is_target:
            this_output_dir = os.path.join(self.output_dir, "target", str(band))
        else:
            this_output_dir = os.path.join(self.output_dir, "input", str(band))
        if not os.path.exists(this_output_dir):
            os.makedirs(this_output_dir, exist_ok=True)
        assert xskew == 0 and yskew == 0, "Geotransform must not have rotation/skew"
        assert (
            xres > 0 and yres < 0
        ), "Geotransform must have positive xres and negative yres"
        xcells, ycells = ds.RasterXSize, ds.RasterYSize
        if is_target:
            no_tiles_width = int(math.ceil(xcells / (tile_size - overlap)))
            no_tiles_height = int(math.ceil(ycells / (tile_size - overlap)))
        else:
            no_tiles_width = tile_ranges[0][2]
            no_tiles_height = tile_ranges[0][3]
            tile_ranges_iter = iter(tile_ranges)
        if ds.RasterCount > 1:
            bnd = ds.GetRasterBand(band)
        else:
            bnd = ds.GetRasterBand(1)
        nodata = bnd.GetNoDataValue()
        proj = ds.GetProjection()
        driver = gdal.GetDriverByName("GTiff")
        for i in range(no_tiles_width):
            for j in range(no_tiles_height):
                if not is_target:
                    tile_range = next(tile_ranges_iter)
                    if len(tile_range) == 0:
                        continue  # Empty tile from target at this location
                    this_xmin = tile_range[0]
                    this_ymax = tile_range[1]
                    xoff = (tile_range[0] - xmin) / xres + i * (tile_size - overlap)
                    yoff = (tile_range[1] - ymax) / yres + j * (tile_size - overlap)
                    this_x_size = (
                        tile_size if xcells - xoff >= tile_size else xcells - xoff
                    )
                    this_y_size = (
                        tile_size if ycells - yoff >= tile_size else ycells - yoff
                    )
                else:
                    xoff = i * (tile_size - overlap)
                    yoff = j * (tile_size - overlap)
                    this_x_size = (
                        tile_size if xcells - xoff >= tile_size else xcells - xoff
                    )
                    this_y_size = (
                        tile_size if ycells - yoff >= tile_size else ycells - yoff
                    )
                # Read from source image at windowed (subset) area
                arr = bnd.ReadAsArray(
                    int(xoff), int(yoff), int(this_x_size), int(this_y_size)
                )
                if arr is None or arr.size == 0 or arr.max() == 0:
                    tile_ranges.append([])
                    continue  # No data found
                this_xmin = xmin + i * (tile_size - overlap) * xres
                this_ymax = ymax + j * (tile_size - overlap) * yres
                out_ds = driver.Create(
                    f"{this_output_dir}/tile_{i}_{j}.tif",
                    this_x_size,
                    this_y_size,
                    1,
                    gdal.GDT_Float32,
                    options=["COMPRESS=LZW"],
                )
                out_band = out_ds.GetRasterBand(1)
                out_band.WriteArray(arr)
                out_band.SetNoDataValue(nodata)
                out_ds.SetProjection(proj)
                out_ds.SetGeoTransform([this_xmin, xres, 0, this_ymax, 0, yres])
                out_ds = None
                if is_target:
                    # Conserve cumulative list of tile ranges for target file
                    tile_ranges.append(
                        (
                            this_xmin,
                            this_ymax,
                            no_tiles_width,
                            no_tiles_height,
                            xres,
                            yres,
                            sr,
                        )
                    )
        return tile_ranges

    def initiate_preprocessing(self):
        """
        After setting up source raster information and parameters, initiate tiling.
        """
        if self.target_file is not None:
            ds = gdal.Open(self.target_file)
            for band in range(1, ds.RasterCount + 1):
                print(f"Processing band {band} of target file")
                if not self.use_downsampled_targets:
                    these_tile_ranges = self.preprocess_band(ds, band, is_target=True)
                else:
                    self.preprocess_single_inputs(ds, band)
        elif self.target_band_files is not None:
            for band in self.target_band_files.keys():
                print(f"Processing band {band} of target file")
                ds = gdal.Open(self.target_band_files[band])
                if not self.use_downsampled_targets:
                    these_tile_ranges = self.preprocess_band(
                        ds, band, is_target=True, tile_ranges=[]
                    )
                else:
                    self.preprocess_single_inputs(ds, band)
        if self.input_file is not None:
            ds = gdal.Open(self.input_file)
            for band in range(1, ds.RasterCount + 1):
                print(f"Processing band {band} of input file")
                these_tile_ranges = self.preprocess_band(
                    ds, band, is_target=False, tile_ranges=these_tile_ranges
                )
        elif self.input_band_files != {}:
            for band in self.input_band_files.keys():
                print(f"Processing band {band} of input file")
                ds = gdal.Open(self.input_band_files[band])
                these_tile_ranges = self.preprocess_band(
                    ds, band, is_target=False, tile_ranges=these_tile_ranges
                )

    @classmethod
    def _match_nulls(
        cls, arr1: np.ndarray, arr2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        With inputs (arr1) as coarser satellite imagery or similar, nulls will be in aerial target images (arr2).

        As such, arr1 will not have nulls and arr2 will.

        Args:
            arr1: Array to be made to have matched nulls/zeros from arr2
            arr2: Array with nulls/zeros to be matched

        Returns:
            Tuple of arrays with matched nulls/zeros
        """
        if arr2.shape != arr1.shape:
            # Use rasterio's nearest-neighbor resampling to match arr2 onto arr1's grid.
            # We don't have georeferencing here, so we treat both arrays as covering the same
            # pixel-space extent and build synthetic transforms accordingly.
            src_h, src_w = arr2.shape
            dst_h, dst_w = arr1.shape

            src_transform = Affine.translation(0, 0) * Affine.scale(1, 1)
            dst_transform = Affine.translation(0, 0) * Affine.scale(
                src_w / dst_w, src_h / dst_h
            )

            arr2_downsampled = np.empty((dst_h, dst_w), dtype=arr2.dtype)
            crs = CRS.from_epsg(4326)
            reproject(
                source=arr2,
                destination=arr2_downsampled,
                src_transform=src_transform,
                src_crs=crs,
                dst_transform=dst_transform,
                dst_crs=crs,
                resampling=Resampling.nearest,
            )
        else:
            arr2_downsampled = arr2
        mask = np.logical_or(arr2_downsampled <= 0, np.isnan(arr2_downsampled))
        arr1[mask] = 0
        arr2 = np.where(np.logical_or(arr2 <= 0, np.isnan(arr2)), 0, arr2)
        return arr1, arr2

    @classmethod
    def _normalize(cls, arr: np.ndarray, gmin: float, gmax: float) -> np.ndarray:
        """
        Helper method to normalize an array to the range [-1, 1].

        Inputs:
            arr (np.ndarray): Array to normalize.
            gmin (float): Global minimum value.
            gmax (float): Global maximum value.

        Returns:
            np.ndarray: Normalized array.
        """
        # Avoid divide-by-zero if constant array
        if gmax == gmin:
            return np.zeros_like(arr)
        return 2 * (arr - gmin) / (gmax - gmin) - 1

    @classmethod
    def _get_as_array(
        cls, raster_file: str, shape: Tuple[int, int], band: int = 1
    ) -> np.ndarray:
        """
        Helper method to read a raster file as an array.

        Inputs:
            raster_file (str): Path to the raster file.
            band (int): Band number to read.

        Returns:
            np.ndarray: Array read from the raster file.
        """
        with gdal.Open(raster_file) as ds:
            band = ds.GetRasterBand(band)
            arr = band.ReadAsArray()
            nodata = band.GetNoDataValue()
        if arr is None:
            raise ValueError(f"Failed to read array from {raster_file}")
        arr = cls._pad(arr=arr, target_shape=shape, fill_value=nodata)
        return arr

    @classmethod
    def _noise(cls, arr, gamma=0.05):
        """
        Apply randomized noise to an image, proportional to gamma input value.

        Inputs:
            gamma(float): magnitude for noise adjustment
        """
        local_noise = (
            np.random.rand(arr.shape[0], arr.shape[1]) * gamma * arr.max()
        )  # Keep scaled to input array values
        arr += local_noise
        return cls._normalize(arr, arr.min(), arr.max())

    @classmethod
    def _pad(
        cls, arr: np.ndarray, target_shape: Tuple[int, int], fill_value: float = -9999
    ) -> np.ndarray:
        """
        Pad an array to match a target shape.  Expects array to match [0,0] coordinates,
        minx and maxy in gdal.

        Inputs:
            arr (np.ndarray): Array to pad.
            target_shape (tuple): Target shape.
            fill_value (float): Default/nodata background to fill

        Returns:
            np.ndarray: Padded array.
        """
        padded = np.full(shape=target_shape, fill_value=fill_value, dtype=arr.dtype)
        padded[: arr.shape[0], : arr.shape[1]] = arr
        return padded

    def to_tensor(
        self,
        save_tensor: bool = True,
        manual_seed: int = None,
        verbose: bool = True,
        band: int = 1,
        using_tiles: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert images in src_dir to tensors.

        Inputs:
            save_tensor (bool): Save tensors to disk.
            manual_seed (int): Manual seed for reproducibility.
            verbose (bool): Print progress.
            band (int): Band to process.
            using_tiles (bool): Whether to use tiles or not.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of tensors - index 0 is training, 1 is input
        """
        targets_dir = os.path.join(self.output_dir, "target", str(band))
        inputs_dir = os.path.join(self.output_dir, "input", str(band))
        # Retrieve targets images (high/goal resolution) from source folder
        src_imgs = glob(os.path.join(targets_dir, "*.tif"))
        target_imgs = []
        target_resolution = (self.tile_size, self.tile_size)
        input_resolution = (
            int(self.tile_size / self.scale_factor),
            int(self.tile_size / self.scale_factor),
        )
        for img in src_imgs:
            if not using_tiles:
                this_img = self._get_as_array(img, target_resolution, band).astype(
                    np.float64
                )
            else:
                this_img = self._get_as_array(img, target_resolution, 1).astype(
                    np.float64
                )
            # Filenames are used to match target and input images
            target_imgs.append((this_img, os.path.basename(img)))
        target_imgs = [
            i for i in target_imgs if np.array(i[0]).max() > 0
        ]  # Allow for zeros/nulls if values are present
        input_im_fs = glob(os.path.join(inputs_dir, "*.tif"))
        input_imgs = []
        for img in input_im_fs:
            if not using_tiles:
                this_img = self._get_as_array(img, input_resolution, band).astype(
                    np.float64
                )
            else:
                this_img = self._get_as_array(img, input_resolution, 1).astype(
                    np.float64
                )
            input_imgs.append((this_img, os.path.basename(img)))
        # Omit any images not sourced in targets folder
        input_imgs = [i for i in input_imgs if i[1] in [j[1] for j in target_imgs]]
        target_img_fs = [i[1] for i in target_imgs]
        # Ensure matching order
        input_imgs = sorted(input_imgs, key=lambda x: target_img_fs.index(x[1]))
        target_imgs = [i[0] for i in target_imgs]
        target_shp = target_imgs[0].shape
        input_imgs = [i[0] for i in input_imgs]
        input_shp = input_imgs[0].shape
        # Individually _normalize images with pair from other set
        new_inputs = []
        new_targets = []
        for arr, target in zip(input_imgs, target_imgs):
            # Interpolate null values from target images to input images
            arr, target = self._match_nulls(arr, target)
            target = match_histograms(target, arr, channel_axis=None)
            target_noise = self._noise(target)
            arr_noise = self._noise(arr)
            arr_flip1 = np.flip(arr_noise, 0)
            arr_flip2 = np.flip(arr_noise, 1)
            tar_flip1 = np.flip(target_noise, 0)
            tar_flip2 = np.flip(target_noise, 1)
            input_arrays = [arr_noise, arr_flip1, arr_flip2]
            target_arrays = [target_noise, tar_flip1, tar_flip2]
            for a in input_arrays:
                # a = np.stack((a, canny(a)))
                new_inputs.append(a)
                new_inputs.append(np.rot90(a, k=1))
                new_inputs.append(np.rot90(a, k=3))
            for a in target_arrays:
                # a = np.stack((a, canny(a)))
                new_targets.append(a)
                new_targets.append(np.rot90(a, k=1))
                new_targets.append(np.rot90(a, k=3))
        x = np.array(
            [
                np.reshape(i, (1, target_shp[0], target_shp[1])).copy()
                for i in target_imgs
            ]
        )
        y = np.array(
            [np.reshape(i, (1, input_shp[0], input_shp[1])).copy() for i in input_imgs]
        )
        # Create random index at length of input, target arrays then apply
        if manual_seed:
            torch.manual_seed(manual_seed)
        z = torch.randperm(x.shape[0])
        x = torch.stack([torch.from_numpy(arr).float() for arr in x])[z]
        y = torch.stack([torch.from_numpy(arr).float() for arr in y])[z]
        if save_tensor:
            save_dir = os.path.join(self.output_dir, "tensors")
            os.makedirs(save_dir, exist_ok=True)
            torch.save(x, os.path.join(save_dir, "x_hr_tensors.pt"))
            torch.save(y, os.path.join(save_dir, "y_lr_tensors.pt"))
        if verbose:
            print(f"Completed {len(x)} batches from {self.output_dir}.")
        if not save_tensor:
            return x, y
        else:
            return None


class ESRGAN:
    """
    Implementation of ESRGAN generator.  Trains both tensor (high resolution) and discriminator.

    Relies on pixel-based loss calculation (L1Loss) and VGG-based perceptual loss.

    Retains two nn.models as class objects: generator and discriminator.

    Generator can be used directly for inference using the geoai.inference.predict_geotiff_superres method.

    Deep learning neural network for image super-resolution. DNN architecture tailored to accommodate
    unique traits of geospatial remotely sensed imagery, especially multi- and hyper-spectral imagery.

    Reference:
        - Song, J., Yi, H., Xu, W., Li, X., Li, B., & Liu, Y. (2023). ESRGAN-DP: Enhanced super-resolution generative adversarial network with adaptive dual perceptual loss. *Heliyon, 9*(4), e15134. https://doi.org/10.1016/j.heliyon.2023.e15134
        - https://github.com/xinntao/ESRGAN
        - DCGAN (PyTorch): https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        - Generative Adversarial Networks (TensorFlow): https://www.tensorflow.org/tutorials/generative/dcgan

    Args:
        scale: Scale factor for upscaling
        band: Band number for the model
        manual_seed: Manual seed for reproducibility
        data_path: Path to the data directory
        model_path: Path to the model directory
    """

    def __init__(
        self,
        scale: int = 4,
        band: int = 1,
        manual_seed: int = None,
        data_path: str = None,
        model_path: str = None,
    ):
        self.scale = scale
        self.band = band
        self.data_path = data_path
        self.model_path = os.path.join(model_path, f"band_{self.band}")
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path, exist_ok=True)
        if manual_seed:
            self.manual_seed = manual_seed
            torch.manual_seed(manual_seed)
        self.metrics = {}
        self.losses = {}
        self.generator = None
        self.discriminator = None

    @classmethod
    def _psnr(cls, mse: torch.tensor) -> torch.tensor:
        """
        Calculate peak signal to noise ratio.

        Inputs:
        - mse: mean squared error

        Returns:
        - psnr: peak signal to noise ratio
        """
        return 10 * torch.log10(1 / mse)

    def train_esrgan(
        self,
        low_res: torch.tensor = None,
        high_res: torch.tensor = None,
        load_tensors: bool = False,
        lr_generator: float = 1e-3,
        lr_discriminator: float = 1e-4,
        lambda_pixel: float = 100.0,
        lambda_vgg: float = 10.0,
        lambda_discriminator: float = 1.0,
        batch_size: int = 8,
        num_epochs: int = 20,
        last_epoch: int = -1,
        use_mp: bool = True,
        detect_anomaly: bool = True,
        input_channels: int = 1,
        output_channels: int = 1,
        nfilters: int = 64,
        nblocks: int = 2,
        ngroups: int = 32,
        warmup_epochs: int = 0,
        validation_split: float = 0.1,
        n_samples: int = None,
    ) -> None:
        """
        ESRGAN-specific training process.

        Creates and/or trains models for generator and discriminator
        in super resolution processing workflow.


        Args:
            low_res (torch.tensor): "Input" image tensor arrays, representing low-res images.
            high_res (torch.tensor): "Target" image tensor arrays, representing images at goal resolution.
            load_tensors (bool): Ignore inputs and load tensors from save path defined in class.
            lr_generator (float): Learning rate for the generator.
            lr_discriminator (float): Learning rate for the discriminator.
            lambda_pixel (float): Weight applied to pixel loss when forming the overall generator loss.
            lambda_vgg (float): Weight applied to VGG perceptual loss when forming the overall generator loss.
            lambda_discriminator (float): Weight applied to adversarial (discriminator) loss when forming the overall generator loss.
            batch_size (int): Number of tiles per batch on generations within epoch.
            num_epochs (int): Number of epochs to train for.
            last_epoch (int): If defined, resume processing from previous iteration.  Zipped model training weights and data (as .pt files) must be defined.
            use_mp (bool): Engage pytorch multiprocessing.
            detect_anomaly (bool): Use pytorch autogradient anomally detection.
            input_channels (int): Number of input channels for the generator.
            output_channels (int): Number of output channels for the generator.
            nfilters (int): Number of filters for the generator.
            nblocks (int): Number of blocks for the generator.
            ngroups (int): Number of groups for the generator.
            warmup_epochs (int): Number of epochs before activating discriminator.
            validation_split (float): Ratio of validation to training images/tiles.
            n_samples (int): Number of samples to use for training. If None, use all samples.

        Raises:
            ValueError

        Examples:
            >>> esrgan = ESRGAN()
            >>> esrgan.train(low_res=torch.tensor(...), high_res=torch.tensor(...))

            >>> from geoai.inference import predict_geotiff
            >>> from geoai.esrgan import ESRGAN
            >>> from os.path import join
            >>> esrgan = ESRGAN(scale=4,
            ...                 band=1,
            ...                 manual_seed=37,
            ...                 data_path='./data',
            ...                 model_path='./models')
            >>> preprocess = ESRGANDataPreprocess(
            ...     output_dir='./data',
            ...     input_band_files={1: 'input_band1.tif'},
            ...     use_downsampled_targets=True,
            ...     scale_factor=4,
            ...     tile_size=128,
            ...     overlap=96,
            ...     batch_size=8,
            ... )
            >>> preprocess.initiate_preprocessing()
            >>> preprocess.to_tensor(save_tensor=True,
            ...                      manual_seed=37,
            ...                      verbose=True,
            ...                      band=1)
            >>> data_path = preprocess.output_dir
            >>> esrgan.train_esrgan(low_res=join(data_path, 'tensors', 'y_lr_tensors.pt'),
            ...                     high_res=join(data_path, 'tensors', 'x_hr_tensors.pt'))
            >>> losses = esrgan.losses
            >>> metrics = esrgan.metrics
            >>> # Access trained models
            >>> generator = esrgan.generator
            >>> discriminator = esrgan.discriminator
            >>> # Plot losses and metrics
            >>> import matplotlib.pyplot as plt
            >>> plt.plot(losses['train'], label='Train Loss')
            >>> plt.plot(losses['val'], label='Val Loss')
            >>> plt.legend()
            >>> plt.show()
            >>> # Perform inference on sample low-res tile 'sample.tif'
            >>> predict_geotiff(
            ...     model=generator,
            ...     input_raster='sample.tif',
            ...     output_raster='sample_hr.tif',
            ...     tile_size=128,
            ...     overlap=96,
            ...     batch_size=8,
            ... )
        """
        # check for either presence of prepared tensors or parameterized inputs
        def _resolve_tensor_input(tensor_or_path, name):
            if isinstance(tensor_or_path, (str, bytes, os.PathLike)):
                return torch.load(tensor_or_path)
            if tensor_or_path is None:
                raise ValueError(f"{name} must not be None.")
            return tensor_or_path

        if not load_tensors and (low_res is None or high_res is None):
            raise ValueError(
                "Either load_tensors must be True or both low_res and high_res must be provided."
            )
        elif load_tensors:
            if not getattr(self, "data_path", None):
                raise ValueError(
                    "load_tensors is True but data_path is not set."
                )
            x_hr_tensor_path = os.path.join(self.data_path, "x_hr_tensors.pt")
            y_lr_tensor_path = os.path.join(self.data_path, "y_lr_tensors.pt")
            if not os.path.exists(x_hr_tensor_path):
                raise ValueError(
                    "load_tensors is True but x_hr_tensors.pt not found in data_path."
                )
            if not os.path.exists(y_lr_tensor_path):
                raise ValueError(
                    "load_tensors is True but y_lr_tensors.pt not found in data_path."
                )
            low_res = torch.load(y_lr_tensor_path)
            high_res = torch.load(x_hr_tensor_path)
        else:
            low_res = _resolve_tensor_input(low_res, "low_res")
            high_res = _resolve_tensor_input(high_res, "high_res")
        device = get_device()
        if use_mp:
            mp.set_start_method("spawn", force=True)
        if detect_anomaly:
            torch.autograd.set_detect_anomaly(True)
        model_path = f"./models/{self.band}"
        os.makedirs(model_path, exist_ok=True)

        # =====================
        # TRAINING SETUP
        # =====================

        gen = ESRGANGenerator(
            in_nc=input_channels,
            out_nc=output_channels,
            nf=nfilters,
            nb=nblocks,
            gc=ngroups,
            scale=self.scale,
        ).to(device)
        self.generator = gen
        dis = Discriminator().to(device)
        self.discriminator = dis

        opt_gen = optim.Adam(gen.parameters(), lr=lr_generator, betas=(0.5, 0.99))
        opt_dis = optim.Adam(dis.parameters(), lr=lr_discriminator, betas=(0.5, 0.99))
        scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_gen,
            mode="max",
            factor=0.5,
            patience=3,
            threshold=1e-3,
            cooldown=1,
            min_lr=1e-7,
        )
        scheduler_D = optim.lr_scheduler.ExponentialLR(opt_dis, gamma=0.9)
        torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(dis.parameters(), 1.0)

        criterion_adv = nn.BCEWithLogitsLoss()
        criterion_pixel = nn.L1Loss()
        criterion_perceptual = VGGPerceptualLoss().to(device)

        # Allow for vgg-specific normalization requirements
        norm_to_vgg = NormalizeToVGG().to(device)

        # Create training validation splits and move input tensors to dataloader
        n_val = max(1, int(len(low_res) * validation_split))
        n_train = len(low_res) - n_val
        train_ds, val_ds = random_split(list(zip(low_res, high_res)), [n_train, n_val])
        if n_samples is not None:
            train_indices = np.arange(min(n_samples, len(train_ds)))
            val_indices = np.arange(min(n_samples, len(val_ds)))
            train_sampler = SubsetRandomSampler(train_indices)
            val_sampler = SubsetRandomSampler(val_indices)
            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                sampler=train_sampler,
                num_workers=4,
                pin_memory=True,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=batch_size,
                sampler=val_sampler,
                num_workers=4,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
            )

        # =====================
        # TRAINING LOOP
        # =====================

        losses = {}
        metrics = {}
        adv_active = False  # only "turn on" adversarial training after initialization

        # Restart a previous epoch from a command line parameter
        if last_epoch > 0:
            pts_file = os.path.join(self.model_path, "generator_checkpoint.pt")
            state_dict = torch.load(pts_file, gen.state_dict())
            result = gen.load_state_dict(state_dict)
            if str(result) != "<All keys matched successfully>":
                raise ValueError("Mismatched weights on generator, check parameters.")
            pts_file = os.path.join(self.model_path, "discriminator_checkpoint.pt")
            state_dict = torch.load(pts_file, dis.state_dict())
            result = dis.load_state_dict(state_dict)
            if str(result) != "<All keys matched successfully>":
                raise ValueError(
                    "Mismatched weights on discriminator, check parameters."
                )

        scaler_G = GradScaler(device.type)
        scaler_D = GradScaler(device.type)

        # increment number of epochs without improvement
        non_improvement_counter = 0
        # Track best epoch improvement (signal to noise ratio)
        best_val_psnr = -1e9

        for epoch in range(last_epoch, num_epochs):

            # "turn on" discriminator/GAN after warmup period
            if epoch >= warmup_epochs:
                adv_active = True
            # set models in training mode
            gen.train()
            dis.train()

            loss_G_avg = 0.0
            loss_D_avg = 0.0
            loss_pixel_avg = 0.0
            loss_vgg_avg = 0.0
            batches = len(train_loader)

            for imgs_lr, imgs_hr in tqdm(train_loader, ascii=True, total=batches):

                imgs_lr = imgs_lr.to(device)
                imgs_hr = imgs_hr.to(device)

                # =====================================================
                # Generator Training
                # =====================================================
                opt_gen.zero_grad()

                gen_hr = gen(imgs_lr)
                # Pixel & perceptual losses
                loss_pixel = criterion_pixel(gen_hr, imgs_hr)
                loss_vgg = criterion_perceptual(
                    norm_to_vgg(gen_hr), norm_to_vgg(imgs_hr)
                )
                # Adversarial loss
                if adv_active:
                    pred_fake = dis(gen_hr)
                    valid = torch.full(pred_fake.shape, 1.0).to(device)
                    loss_G_adv = criterion_adv(pred_fake, valid)
                    loss_G = (
                        (loss_pixel * lambda_pixel)
                        + (loss_vgg * lambda_vgg)
                        + (loss_G_adv * lambda_discriminator)
                    )
                else:
                    # initial iterations measuring pixel and perceptual loss only
                    loss_G = (loss_pixel * lambda_pixel) + (loss_vgg * lambda_vgg)

                scaler_G.scale(loss_G).backward()
                scaler_G.unscale_(opt_gen)
                clip_grad_norm_(gen.parameters(), 1.0)
                scaler_G.step(opt_gen)
                scaler_G.update()

                # =====================================================
                # Discriminator Training
                # =====================================================
                if adv_active:
                    opt_dis.zero_grad()

                    # Real images
                    pred_real = dis(imgs_hr)
                    valid = torch.full(pred_real.shape, 1.0).to(device)
                    loss_real = criterion_adv(pred_real, valid)

                    # Fake/generated images
                    pred_fake = dis(gen_hr.detach())
                    fake = torch.full(pred_fake.shape, 0.0).to(device)
                    loss_fake = criterion_adv(pred_fake, fake)

                    loss_D = 0.5 * (loss_real + loss_fake)
                    scaler_D.scale(loss_D).backward()
                    scaler_D.unscale_(opt_dis)
                    clip_grad_norm_(dis.parameters(), 1.0)
                    scaler_D.step(opt_dis)
                    scaler_D.update()
                else:
                    loss_D = torch.tensor(0.0)

                loss_G_avg += loss_G.item()
                loss_pixel_avg += loss_pixel.item()
                loss_vgg_avg += loss_vgg.item()
                loss_D_avg += loss_D.item()

            # =====================================================
            # Validation Check and Statistics
            # =====================================================
            gen.eval()
            dis.eval()
            val_psnrs = []
            val_vggs = []
            val_mses = []
            with torch.no_grad():
                for imgs_lr, imgs_hr in val_loader:
                    imgs_lr = imgs_lr.to(device)
                    imgs_hr = imgs_hr.to(device)
                    sr = gen(imgs_lr)
                    mse = F.mse_loss(sr, imgs_hr, reduction="mean")
                    val_mses.append(mse)
                    val_psnrs.append(self._psnr(torch.tensor(mse)))
                    # perceptual/VGG loss
                    vgg_loss = criterion_perceptual(
                        norm_to_vgg(sr), norm_to_vgg(imgs_hr)
                    ).item()
                    val_vggs.append(vgg_loss)

            mean_val_psnr = sum(val_psnrs) / len(val_psnrs)
            mean_val_vgg = sum(val_vggs) / len(val_vggs)
            mean_val_mse = sum(val_mses) / len(val_mses)
            print(
                f"Validation — PSNR: {mean_val_psnr:.3f}, VGG: {mean_val_vgg:.6f}, MSE: {mean_val_mse:.6f}"
            )
            if mean_val_psnr > best_val_psnr:
                best_val_psnr = mean_val_psnr
                torch.save(
                    gen.state_dict(), os.path.join(model_path, "best_generator.pt")
                )
                torch.save(
                    dis.state_dict(), os.path.join(model_path, "best_discriminator.pt")
                )
            else:
                non_improvement_counter += 1
            # =====================================================
            # Step learning-rate schedulers
            # =====================================================
            scheduler_G.step(mean_val_psnr)
            if (
                non_improvement_counter >= 2
            ):  # 2 consecutive epoch(s) without improvement
                non_improvement_counter = 0
                scheduler_D.step()

            # =====================================================
            # Logging
            # =====================================================
            print(
                f"[Epoch {epoch+1}/{num_epochs}]  "
                f"Gen: {loss_G_avg/batches:.4f}  "
                f"Dis: {loss_D_avg/batches:.4f} "
                f"Pix: {loss_pixel_avg/batches:.4f} "
                f"VGG: {loss_vgg_avg/batches:.4f}"
            )
            if not "Gen" in losses:
                losses["Gen"] = []
            if not "Dis" in losses:
                losses["Dis"] = []
            if not "Pix" in losses:
                losses["Pix"] = []
            if not "VGG" in losses:
                losses["VGG"] = []
            if not "mse" in metrics:
                metrics["mse"] = []
            if not "msnr" in metrics:
                metrics["msnr"] = []
            losses["Gen"].append(loss_G_avg / batches)
            losses["Dis"].append(loss_D_avg / batches)
            losses["Pix"].append(loss_pixel_avg / batches)
            losses["VGG"].append(loss_vgg_avg / batches)
            metrics["mse"].append(mean_val_mse.item())
            metrics["msnr"].append(mean_val_psnr.item())

            self.losses = losses
            self.metrics = metrics

            # =====================================================
            # Save model weights
            # =====================================================

            torch.save(
                gen.state_dict(),
                os.path.join(self.model_path, "generator_checkpoint.pt"),
            )
            torch.save(
                dis.state_dict(),
                os.path.join(self.model_path, "discriminator_checkpoint.pt"),
            )
