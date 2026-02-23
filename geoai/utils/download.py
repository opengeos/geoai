"""File download utilities."""

import os
from typing import Optional

import requests

__all__ = ["download_file", "download_model_from_hf"]


def download_file(
    url: str,
    output_path: Optional[str] = None,
    overwrite: bool = False,
    unzip: bool = True,
) -> str:
    """
    Download a file from a given URL with a progress bar.
    Optionally unzip the file if it's a ZIP archive.

    Args:
        url (str): The URL of the file to download.
        output_path (str, optional): The path where the downloaded file will be saved.
            If not provided, the filename from the URL will be used.
        overwrite (bool, optional): Whether to overwrite the file if it already exists.
        unzip (bool, optional): Whether to unzip the file if it is a ZIP archive.

    Returns:
        str: The path to the downloaded file or the extracted directory.
    """

    import zipfile

    from tqdm import tqdm

    if output_path is None:
        output_path = os.path.basename(url)

    if os.path.exists(output_path) and not overwrite:
        print(f"File already exists: {output_path}")
    else:
        # Download the file with a progress bar
        response = requests.get(url, stream=True, timeout=50)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with (
            open(output_path, "wb") as file,
            tqdm(
                desc=f"Downloading {os.path.basename(output_path)}",
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar,
        ):
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
                    progress_bar.update(len(chunk))

    # If the file is a ZIP archive and unzip is True
    if unzip and zipfile.is_zipfile(output_path):
        extract_dir = os.path.splitext(output_path)[0]
        if not os.path.exists(extract_dir) or overwrite:
            with zipfile.ZipFile(output_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"Extracted to: {extract_dir}")
        return extract_dir

    return output_path


def download_model_from_hf(model_path: str, repo_id: Optional[str] = None) -> str:
    """
    Download the object detection model from Hugging Face.

    Args:
        model_path: Path to the model file.
        repo_id: Hugging Face repository ID.

    Returns:
        Path to the downloaded model file
    """
    from huggingface_hub import hf_hub_download

    try:

        # Define the repository ID and model filename
        if repo_id is None:
            print(
                "Repo is not specified, using default Hugging Face repo_id: giswqs/geoai"
            )
            repo_id = "giswqs/geoai"

        # Download the model
        model_path = hf_hub_download(repo_id=repo_id, filename=model_path)
        print(f"Model downloaded to: {model_path}")

        return model_path

    except Exception as e:
        print(f"Error downloading model from Hugging Face: {e}")
        print("Please specify a local model path or ensure internet connectivity.")
        raise
