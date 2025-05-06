"""
The SamGeo class provides an interface for segmenting geospatial data using the Segment Anything Model (SAM).
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from leafmap import array_to_image, blend_images
from PIL import Image
from transformers import SamModel, SamProcessor, pipeline

from .utils import *


class SamGeo:
    """The main class for segmenting geospatial data with the Segment Anything Model (SAM). See
    https://huggingface.co/docs/transformers/main/en/model_doc/sam for details.
    """

    def __init__(
        self,
        model: str = "facebook/sam-vit-huge",
        automatic: bool = True,
        device: Optional[Union[str, int]] = None,
        sam_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the class.

        Args:
            model (str, optional): The model type, such as "facebook/sam-vit-huge", "facebook/sam-vit-large", or "facebook/sam-vit-base".
                Defaults to 'facebook/sam-vit-huge'. See https://bit.ly/3VrpxUh for more details.
            automatic (bool, optional): Whether to use the automatic mask generator or input prompts. Defaults to True.
                The automatic mask generator will segment the entire image, while the input prompts will segment selected objects.
            device (Union[str, int], optional): The device to use. It can be one of the following: 'cpu', 'cuda', or an integer
                representing the CUDA device index. Defaults to None, which will use 'cuda' if available.
            sam_kwargs (Dict[str, Any], optional): Optional arguments for fine-tuning the SAM model. Defaults to None.
            kwargs (Any): Other arguments for the automatic mask generator.
        """

        self.model = model
        self.model_version = "sam"

        self.sam_kwargs = sam_kwargs  # Optional arguments for fine-tuning the SAM model
        self.source = None  # Store the input image path
        self.image = None  # Store the input image as a numpy array
        self.embeddings = None  # Store the image embeddings
        # Store the masks as a list of dictionaries. Each mask is a dictionary
        # containing segmentation, area, bbox, predicted_iou, point_coords, stability_score, and crop_box
        self.masks = None
        self.objects = None  # Store the mask objects as a numpy array
        # Store the annotations (objects with random color) as a numpy array.
        self.annotations = None

        # Store the predicted masks, iou_predictions, and low_res_masks
        self.prediction = None
        self.scores = None
        self.logits = None

        # Build the SAM model
        sam_kwargs = self.sam_kwargs if self.sam_kwargs is not None else {}

        if automatic:
            # Use cuda if available
            if device is None:
                device = 0 if torch.cuda.is_available() else -1
            if device >= 0:
                torch.cuda.empty_cache()
            self.device = device

            self.mask_generator = pipeline(
                task="mask-generation",
                model=model,
                device=device,
                **kwargs,
            )

        else:
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                self.device = device

                self.predictor = SamModel.from_pretrained("facebook/sam-vit-huge").to(
                    device
                )
                self.processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    def generate(
        self,
        source: Union[str, np.ndarray],
        output: Optional[str] = None,
        foreground: bool = True,
        erosion_kernel: Optional[Tuple[int, int]] = None,
        mask_multiplier: int = 255,
        unique: bool = True,
        min_size: int = 0,
        max_size: Optional[int] = None,
        output_args: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Generate masks for the input image.

        Args:
            source (Union[str, np.ndarray]): The path to the input image or the input image as a numpy array.
            output (Optional[str], optional): The path to the output image. Defaults to None.
            foreground (bool, optional): Whether to generate the foreground mask. Defaults to True.
            erosion_kernel (Optional[Tuple[int, int]], optional): The erosion kernel for filtering object masks and extracting borders.
                For example, (3, 3) or (5, 5). Set to None to disable it. Defaults to None.
            mask_multiplier (int, optional): The mask multiplier for the output mask, which is usually a binary mask [0, 1].
                You can use this parameter to scale the mask to a larger range, for example [0, 255]. Defaults to 255.
            unique (bool, optional): Whether to assign a unique value to each object. Defaults to True.
                The unique value increases from 1 to the number of objects. The larger the number, the larger the object area.
            min_size (int, optional): The minimum size of the objects. Defaults to 0.
            max_size (Optional[int], optional): The maximum size of the objects. Defaults to None.
            output_args (Optional[Dict[str, Any]], optional): Additional arguments for saving the output. Defaults to None.
            **kwargs (Any): Other arguments for the mask generator.

        Raises:
            ValueError: If the input source is not a valid path or numpy array.
        """

        if isinstance(source, str):
            if source.startswith("http"):
                source = download_file(source)

            if not os.path.exists(source):
                raise ValueError(f"Input path {source} does not exist.")

            image = cv2.imread(source)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(source, np.ndarray):
            image = source
            source = None
        else:
            raise ValueError("Input source must be either a path or a numpy array.")

        if output_args is None:
            output_args = {}

        self.source = source  # Store the input image path
        self.image = image  # Store the input image as a numpy array
        mask_generator = self.mask_generator  # The automatic mask generator
        # masks = mask_generator.generate(image)  # Segment the input image
        result = mask_generator(source, **kwargs)
        masks = result["masks"] if "masks" in result else result  # Get the masks
        scores = result["scores"] if "scores" in result else None  # Get the scores

        # format the masks as a list of dictionaries, similar to the output of SAM.
        formatted_masks = []
        for mask, score in zip(masks, scores):
            area = int(np.sum(mask))  # number of True pixels
            formatted_masks.append(
                {
                    "segmentation": mask,
                    "area": area,
                    "score": float(score),  # ensure it's a native Python float
                }
            )

        self.output = result  # Store the result
        self.masks = formatted_masks  # Store the masks as a list of dictionaries
        self.batch = False
        # self.scores = scores  # Store the scores
        self._min_size = min_size
        self._max_size = max_size

        # Save the masks to the output path. The output is either a binary mask or a mask of objects with unique values.
        self.save_masks(
            output,
            foreground,
            unique,
            erosion_kernel,
            mask_multiplier,
            min_size,
            max_size,
            **output_args,
        )

    def generate_batch(
        self,
        inputs: List[Union[str, np.ndarray]],
        output_dir: Optional[str] = None,
        suffix: str = "_masks",
        foreground: bool = True,
        erosion_kernel: Optional[Tuple[int, int]] = None,
        mask_multiplier: int = 255,
        unique: bool = True,
        min_size: int = 0,
        max_size: Optional[int] = None,
        output_args: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Generate masks for a batch of input images.

        Args:
            inputs (List[Union[str, np.ndarray]]): A list of paths to input images or numpy arrays representing the images.
            output_dir (Optional[str], optional): The directory to save the output masks. Defaults to the current working directory.
            suffix (str, optional): The suffix to append to the output filenames. Defaults to "_masks".
            foreground (bool, optional): Whether to generate the foreground mask. Defaults to True.
            erosion_kernel (Optional[Tuple[int, int]], optional): The erosion kernel for filtering object masks and extracting borders.
                For example, (3, 3) or (5, 5). Set to None to disable it. Defaults to None.
            mask_multiplier (int, optional): The mask multiplier for the output mask, which is usually a binary mask [0, 1].
                You can use this parameter to scale the mask to a larger range, for example [0, 255]. Defaults to 255.
            unique (bool, optional): Whether to assign a unique value to each object. Defaults to True.
                The unique value increases from 1 to the number of objects. The larger the number, the larger the object area.
            min_size (int, optional): The minimum size of the objects. Defaults to 0.
            max_size (Optional[int], optional): The maximum size of the objects. Defaults to None.
            output_args (Optional[Dict[str, Any]], optional): Additional arguments for saving the output. Defaults to None.
            **kwargs (Any): Other arguments for the mask generator.

        Raises:
            ValueError: If the input list is empty or contains invalid paths.
        """

        mask_generator = self.mask_generator  # The automatic mask generator
        outputs = mask_generator(inputs, **kwargs)

        if output_args is None:
            output_args = {}

        if output_dir is None:
            output_dir = os.getcwd()

        for index, result in enumerate(outputs):

            basename = os.path.basename(inputs[index])
            file_ext = os.path.splitext(basename)[1]
            filename = f"{os.path.splitext(basename)[0]}{suffix}{file_ext}"
            filepath = os.path.join(output_dir, filename)

            masks = result["masks"] if "masks" in result else result  # Get the masks
            scores = result["scores"] if "scores" in result else None  # Get the scores

            # format the masks as a list of dictionaries, similar to the output of SAM.
            formatted_masks = []
            for mask, score in zip(masks, scores):
                area = int(np.sum(mask))  # number of True pixels
                formatted_masks.append(
                    {
                        "segmentation": mask,
                        "area": area,
                        "score": float(score),  # ensure it's a native Python float
                    }
                )

            self.source = inputs[index]  # Store the input image path
            self.output = result  # Store the result
            self.masks = formatted_masks  # Store the masks as a list of dictionaries
            # self.scores = scores  # Store the scores
            self._min_size = min_size
            self._max_size = max_size

            # Save the masks to the output path. The output is either a binary mask or a mask of objects with unique values.
            self.save_masks(
                filepath,
                foreground,
                unique,
                erosion_kernel,
                mask_multiplier,
                min_size,
                max_size,
                **output_args,
            )

    def save_masks(
        self,
        output: Optional[str] = None,
        foreground: bool = True,
        unique: bool = True,
        erosion_kernel: Optional[Tuple[int, int]] = None,
        mask_multiplier: int = 255,
        min_size: int = 0,
        max_size: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        Save the masks to the output path. The output is either a binary mask or a mask of objects with unique values.

        Args:
            output (Optional[str], optional): The path to the output image. Defaults to None, saving the masks to `SamGeo.objects`.
            foreground (bool, optional): Whether to generate the foreground mask. Defaults to True.
            unique (bool, optional): Whether to assign a unique value to each object. Defaults to True.
            erosion_kernel (Optional[Tuple[int, int]], optional): The erosion kernel for filtering object masks and extracting borders.
                For example, (3, 3) or (5, 5). Set to None to disable it. Defaults to None.
            mask_multiplier (int, optional): The mask multiplier for the output mask, which is usually a binary mask [0, 1].
                You can use this parameter to scale the mask to a larger range, for example [0, 255]. Defaults to 255.
            min_size (int, optional): The minimum size of the objects. Defaults to 0.
            max_size (Optional[int], optional): The maximum size of the objects. Defaults to None.
            **kwargs (Any): Other arguments for `array_to_image()`.

        Raises:
            ValueError: If no masks are found or if `generate()` has not been run.
        """

        if self.masks is None:
            raise ValueError("No masks found. Please run generate() first.")

        if self.image is None:
            (
                h,
                w,
            ) = self.masks[
                0
            ]["segmentation"].shape
        else:
            h, w, _ = self.image.shape
        masks = self.masks

        # Set output image data type based on the number of objects
        if len(masks) < 255:
            dtype = np.uint8
        elif len(masks) < 65535:
            dtype = np.uint16
        else:
            dtype = np.uint32

        # Generate a mask of objects with unique values
        if unique:
            # Sort the masks by area in descending order
            sorted_masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)

            # Create an output image with the same size as the input image
            objects = np.zeros(
                (
                    sorted_masks[0]["segmentation"].shape[0],
                    sorted_masks[0]["segmentation"].shape[1],
                )
            )
            # Assign a unique value to each object
            count = len(sorted_masks)
            for index, ann in enumerate(sorted_masks):
                m = ann["segmentation"]
                if min_size > 0 and ann["area"] < min_size:
                    continue
                if max_size is not None and ann["area"] > max_size:
                    continue
                objects[m] = count - index

        # Generate a binary mask
        else:
            if foreground:  # Extract foreground objects only
                resulting_mask = np.zeros((h, w), dtype=dtype)
            else:
                resulting_mask = np.ones((h, w), dtype=dtype)
            resulting_borders = np.zeros((h, w), dtype=dtype)

            for m in masks:
                if min_size > 0 and m["area"] < min_size:
                    continue
                if max_size is not None and m["area"] > max_size:
                    continue
                mask = (m["segmentation"] > 0).astype(dtype)
                resulting_mask += mask

                # Apply erosion to the mask
                if erosion_kernel is not None:
                    mask_erode = cv2.erode(mask, erosion_kernel, iterations=1)
                    mask_erode = (mask_erode > 0).astype(dtype)
                    edge_mask = mask - mask_erode
                    resulting_borders += edge_mask

            resulting_mask = (resulting_mask > 0).astype(dtype)
            resulting_borders = (resulting_borders > 0).astype(dtype)
            objects = resulting_mask - resulting_borders
            objects = objects * mask_multiplier

        objects = objects.astype(dtype)
        self.objects = objects

        if output is not None:  # Save the output image
            array_to_image(self.objects, output, self.source, **kwargs)

    def show_masks(
        self,
        figsize: Tuple[int, int] = (12, 10),
        cmap: str = "binary_r",
        axis: str = "off",
        foreground: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Display the binary mask or the mask of objects with unique values.

        Args:
            figsize (Tuple[int, int], optional): The figure size. Defaults to (12, 10).
            cmap (str, optional): The colormap to use for displaying the mask. Defaults to "binary_r".
            axis (str, optional): Whether to show the axis. Defaults to "off".
            foreground (bool, optional): Whether to show the foreground mask only. Defaults to True.
            **kwargs (Any): Additional arguments for the `save_masks()` method.

        Raises:
            ValueError: If no masks are available and `save_masks()` cannot generate them.
        """
        import matplotlib.pyplot as plt

        if self.batch:
            self.objects = cv2.imread(self.masks)
        else:
            if self.objects is None:
                self.save_masks(foreground=foreground, **kwargs)

        plt.figure(figsize=figsize)
        plt.imshow(self.objects, cmap=cmap)
        plt.axis(axis)
        plt.show()

    def show_anns(
        self,
        figsize: Tuple[int, int] = (12, 10),
        axis: str = "off",
        alpha: float = 0.35,
        output: Optional[str] = None,
        blend: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Show the annotations (objects with random color) on the input image.

        Args:
            figsize (Tuple[int, int], optional): The figure size. Defaults to (12, 10).
            axis (str, optional): Whether to show the axis. Defaults to "off".
            alpha (float, optional): The alpha value for the annotations. Defaults to 0.35.
            output (Optional[str], optional): The path to the output image. Defaults to None.
            blend (bool, optional): Whether to show the input image blended with annotations. Defaults to True.
            **kwargs (Any): Additional arguments for saving the output image.

        Raises:
            ValueError: If the input image or annotations are not available.
        """

        import matplotlib.pyplot as plt

        anns = self.masks

        if self.image is None:
            print("Please run generate() first.")
            return

        if anns is None or len(anns) == 0:
            return

        plt.figure(figsize=figsize)
        plt.imshow(self.image)

        sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)

        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones(
            (
                sorted_anns[0]["segmentation"].shape[0],
                sorted_anns[0]["segmentation"].shape[1],
                4,
            )
        )
        img[:, :, 3] = 0
        for ann in sorted_anns:
            if hasattr(self, "_min_size") and (ann["area"] < self._min_size):
                continue
            if (
                hasattr(self, "_max_size")
                and isinstance(self._max_size, int)
                and ann["area"] > self._max_size
            ):
                continue
            m = ann["segmentation"]
            color_mask = np.concatenate([np.random.random(3), [alpha]])
            img[m] = color_mask
        ax.imshow(img)

        # if "dpi" not in kwargs:
        #     kwargs["dpi"] = 100

        # if "bbox_inches" not in kwargs:
        #     kwargs["bbox_inches"] = "tight"

        plt.axis(axis)

        self.annotations = (img[:, :, 0:3] * 255).astype(np.uint8)

        if output is not None:
            if blend:
                array = blend_images(
                    self.annotations, self.image, alpha=alpha, show=False
                )
            else:
                array = self.annotations
            array_to_image(array, output, self.source, **kwargs)

    def set_image(self, image: Union[str, np.ndarray], **kwargs: Any) -> None:
        """
        Set the input image as a numpy array.

        Args:
            image (Union[str, np.ndarray]): The input image, either as a file path (string) or a numpy array.
            **kwargs (Any): Additional arguments for the image processor.

        Raises:
            ValueError: If the input image path does not exist.
        """
        if isinstance(image, str):
            if image.startswith("http"):
                image = download_file(image)

            if not os.path.exists(image):
                raise ValueError(f"Input path {image} does not exist.")

            self.source = image

            image = Image.open(image).convert("RGB")
            self.image = image

        inputs = self.processor(image, return_tensors="pt").to(self.device)
        self.embeddings = self.predictor.get_image_embeddings(
            inputs["pixel_values"], **kwargs
        )

    def save_prediction(
        self,
        output: str,
        index: Optional[int] = None,
        mask_multiplier: int = 255,
        dtype: np.dtype = np.float32,
        vector: Optional[str] = None,
        simplify_tolerance: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        Save the predicted mask to the output path.

        Args:
            output (str): The path to the output image.
            index (Optional[int], optional): The index of the mask to save. Defaults to None,
                which will save the mask with the highest score.
            mask_multiplier (int, optional): The mask multiplier for the output mask, which is usually a binary mask [0, 1].
                Defaults to 255.
            dtype (np.dtype, optional): The data type of the output image. Defaults to np.float32.
            vector (Optional[str], optional): The path to the output vector file. Defaults to None.
            simplify_tolerance (Optional[float], optional): The maximum allowed geometry displacement.
                The higher this value, the smaller the number of vertices in the resulting geometry. Defaults to None.
            **kwargs (Any): Additional arguments for saving the output image.

        Raises:
            ValueError: If no predictions are found.
        """
        if self.scores is None:
            raise ValueError("No predictions found. Please run predict() first.")

        if index is None:
            index = self.scores.argmax(axis=0)

        array = self.masks[index] * mask_multiplier
        self.prediction = array
        array_to_image(array, output, self.source, dtype=dtype, **kwargs)

        if vector is not None:
            raster_to_vector(output, vector, simplify_tolerance=simplify_tolerance)

    def predict(
        self,
        point_coords=None,
        point_labels=None,
        boxes=None,
        point_crs=None,
        mask_input=None,
        multimask_output=True,
        return_logits=False,
        output=None,
        index=None,
        mask_multiplier=255,
        dtype="float32",
        return_results=False,
        **kwargs,
    ):
        """Predict masks for the given input prompts, using the currently set image.

        Args:
            point_coords (str | dict | list | np.ndarray, optional): A Nx2 array of point prompts to the
                model. Each point is in (X,Y) in pixels. It can be a path to a vector file, a GeoJSON
                dictionary, a list of coordinates [lon, lat], or a numpy array. Defaults to None.
            point_labels (list | int | np.ndarray, optional): A length N array of labels for the
                point prompts. 1 indicates a foreground point and 0 indicates a background point.
            point_crs (str, optional): The coordinate reference system (CRS) of the point prompts.
            boxes (list | np.ndarray, optional): A length 4 array given a box prompt to the
                model, in XYXY format.
            mask_input (np.ndarray, optional): A low resolution mask input to the model, typically
                coming from a previous prediction iteration. Has form 1xHxW, where for SAM, H=W=256.
                multimask_output (bool, optional): If true, the model will return three masks.
                For ambiguous input prompts (such as a single click), this will often
                produce better masks than a single prediction. If only a single
                mask is needed, the model's predicted quality score can be used
                to select the best mask. For non-ambiguous prompts, such as multiple
                input prompts, multimask_output=False can give better results.
            return_logits (bool, optional): If true, returns un-thresholded masks logits
                instead of a binary mask.
            output (str, optional): The path to the output image. Defaults to None.
            index (index, optional): The index of the mask to save. Defaults to None,
                which will save the mask with the highest score.
            mask_multiplier (int, optional): The mask multiplier for the output mask, which is usually a binary mask [0, 1].
            dtype (np.dtype, optional): The data type of the output image. Defaults to np.float32.
            return_results (bool, optional): Whether to return the predicted masks, scores, and logits. Defaults to False.

        """
        out_of_bounds = []

        if isinstance(boxes, str):
            gdf = gpd.read_file(boxes)
            if gdf.crs is not None:
                gdf = gdf.to_crs("epsg:4326")
            boxes = gdf.geometry.bounds.values.tolist()
        elif isinstance(boxes, dict):
            import json

            geojson = json.dumps(boxes)
            gdf = gpd.read_file(geojson, driver="GeoJSON")
            boxes = gdf.geometry.bounds.values.tolist()

        if isinstance(point_coords, str):
            point_coords = vector_to_geojson(point_coords)

        if isinstance(point_coords, dict):
            point_coords = geojson_to_coords(point_coords)

        if hasattr(self, "point_coords"):
            point_coords = self.point_coords

        if hasattr(self, "point_labels"):
            point_labels = self.point_labels

        if (point_crs is not None) and (point_coords is not None):
            point_coords, out_of_bounds = coords_to_xy(
                self.source, point_coords, point_crs, return_out_of_bounds=True
            )

        if isinstance(point_coords, list):
            point_coords = np.array(point_coords)

        if point_coords is not None:
            if point_labels is None:
                point_labels = [1] * len(point_coords)
            elif isinstance(point_labels, int):
                point_labels = [point_labels] * len(point_coords)

        if isinstance(point_labels, list):
            if len(point_labels) != len(point_coords):
                if len(point_labels) == 1:
                    point_labels = point_labels * len(point_coords)
                elif len(out_of_bounds) > 0:
                    print(f"Removing {len(out_of_bounds)} out-of-bound points.")
                    point_labels_new = []
                    for i, p in enumerate(point_labels):
                        if i not in out_of_bounds:
                            point_labels_new.append(p)
                    point_labels = point_labels_new
                else:
                    raise ValueError(
                        "The length of point_labels must be equal to the length of point_coords."
                    )
            point_labels = np.array(point_labels)

        predictor = self.predictor

        input_boxes = None
        if isinstance(boxes, list) and (point_crs is not None):
            coords = bbox_to_xy(self.source, boxes, point_crs)
            input_boxes = np.array(coords)
            if isinstance(coords[0], int):
                input_boxes = input_boxes[None, :]
            else:
                input_boxes = torch.tensor(input_boxes, device=self.device)
                input_boxes = predictor.transform.apply_boxes_torch(
                    input_boxes, self.image.shape[:2]
                )
        elif isinstance(boxes, list) and (point_crs is None):
            input_boxes = np.array(boxes)
            if isinstance(boxes[0], int):
                input_boxes = input_boxes[None, :]

        self.boxes = input_boxes
        self.point_coords = point_coords
        self.point_labels = point_labels

        if input_boxes is not None:
            input_boxes = [input_boxes]

        if point_coords is not None:
            point_coords = [[point_coords]]
            point_labels = [[point_labels]]

        inputs = self.processor(
            self.image,
            input_points=point_coords,
            # input_labels=point_labels,
            input_boxes=input_boxes,
            return_tensors="pt",
            **kwargs,
        ).to(self.device)

        inputs.pop("pixel_values", None)
        inputs.update({"image_embeddings": self.embeddings})

        with torch.no_grad():
            outputs = self.predictor(**inputs)

        # https://huggingface.co/docs/transformers/en/model_doc/sam#transformers.SamImageProcessor.post_process_masks
        self.masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        self.scores = outputs.iou_scores

        # if (
        #     boxes is None
        #     or (len(boxes) == 1)
        #     or (len(boxes) == 4 and isinstance(boxes[0], float))
        # ):
        #     if isinstance(boxes, list) and isinstance(boxes[0], list):
        #         boxes = boxes[0]
        #     masks, scores, logits = predictor.predict(
        #         point_coords,
        #         point_labels,
        #         input_boxes,
        #         mask_input,
        #         multimask_output,
        #         return_logits,
        #     )
        # else:
        #     masks, scores, logits = predictor.predict_torch(
        #         point_coords=point_coords,
        #         point_labels=point_coords,
        #         boxes=input_boxes,
        #         multimask_output=True,
        #     )

        # self.masks = masks
        # self.scores = scores
        # self.logits = logits

        # if output is not None:
        #     if boxes is None or (not isinstance(boxes[0], list)):
        #         self.save_prediction(output, index, mask_multiplier, dtype, **kwargs)
        #     else:
        #         self.tensor_to_numpy(
        #             index, output, mask_multiplier, dtype, save_args=kwargs
        #         )

        # if return_results:
        #     return masks, scores, logits

    def tensor_to_numpy(
        self,
        index: Optional[int] = None,
        output: Optional[str] = None,
        mask_multiplier: int = 255,
        dtype: Union[str, np.dtype] = "uint8",
        save_args: Optional[Dict[str, Any]] = None,
    ) -> Optional[np.ndarray]:
        """
        Convert the predicted masks from tensors to numpy arrays.

        Args:
            index (Optional[int], optional): The index of the mask to save. Defaults to None,
                which will save the mask with the highest score.
            output (Optional[str], optional): The path to the output image. Defaults to None.
            mask_multiplier (int, optional): The mask multiplier for the output mask, which is usually a binary mask [0, 1].
                Defaults to 255.
            dtype (Union[str, np.dtype], optional): The data type of the output image. Defaults to "uint8".
            save_args (Optional[Dict[str, Any]], optional): Optional arguments for saving the output image. Defaults to None.

        Returns:
            Optional[np.ndarray]: The predicted mask as a numpy array if `output` is None. Otherwise, saves the mask to the specified path.

        Raises:
            ValueError: If no objects are found in the image or if the masks are not available.
        """

        if save_args is None:
            save_args = {}

        if self.masks is None:
            raise ValueError("No masks found. Please run the prediction method first.")

        boxes = self.boxes
        masks = self.masks

        image_pil = self.image
        image_np = np.array(image_pil)

        if index is None:
            index = 1

        masks = masks[:, index, :, :]
        masks = masks.squeeze(1)

        if boxes is None or (len(boxes) == 0):  # No "object" instances found
            print("No objects found in the image.")
            return
        else:
            # Create an empty image to store the mask overlays
            mask_overlay = np.zeros_like(
                image_np[..., 0], dtype=dtype
            )  # Adjusted for single channel

            for i, (box, mask) in enumerate(zip(boxes, masks)):
                # Convert tensor to numpy array if necessary and ensure it contains integers
                if isinstance(mask, torch.Tensor):
                    mask = (
                        mask.cpu().numpy().astype(dtype)
                    )  # If mask is on GPU, use .cpu() before .numpy()
                mask_overlay += ((mask > 0) * (i + 1)).astype(
                    dtype
                )  # Assign a unique value for each mask

            # Normalize mask_overlay to be in [0, 255]
            mask_overlay = (
                mask_overlay > 0
            ) * mask_multiplier  # Binary mask in [0, 255]

        if output is not None:
            array_to_image(mask_overlay, output, self.source, dtype=dtype, **save_args)
        else:
            return mask_overlay
