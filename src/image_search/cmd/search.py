"""
Search command for the image-search CLI.
"""

import hashlib
import os
from pathlib import Path
from typing import Annotated

import chromadb
import gradio
import numpy as np
import PIL
import platformdirs
import typer
from chromadb.utils.embedding_functions.open_clip_embedding_function import (
    OpenCLIPEmbeddingFunction,
)
from diskcache import Cache
from rich import print  # pylint: disable=redefined-builtin
from tqdm.auto import tqdm


def main() -> None:
    """Launch the main application."""
    typer.run(_cli)


def _cli(
    directory: Annotated[
        Path, typer.Argument(help="Directory containing images to search.")
    ],
) -> None:
    """Search for images in a directory using OpenCLIP embeddings."""
    # Create a ChromaDB instance and image collection
    print("Setting up ChromaDB...")
    db = chromadb.Client()
    collection = db.create_collection(
        "images",
        embedding_function=OpenCLIPEmbeddingFunction(device="mps"),
    )

    # Find all of the images in the directory, searching recursively
    print("Identifying images...")
    image_paths = [
        path.resolve()
        for path in directory.glob("**/*")
        if path.suffix in {".jpg", ".jpeg", ".png"}
    ]

    # Initialize a cache to store the embeddings across invocations of this command
    cache_directory = platformdirs.user_cache_dir(
        appname="image-search",
        appauthor="connorbrinton",
        ensure_exists=True,
    )
    cache = Cache(os.path.join(cache_directory, "embeddings.db"))

    # Index all of the images in the directory
    for image_path in tqdm(image_paths, desc="Indexing images"):
        # Load the contents of the image file into memory
        with image_path.open("rb") as image_file:
            image_bytes = image_file.read()

        # Hash the contents of the image file into a key for our cache
        image_hash = hashlib.blake2b(image_bytes).digest()

        # Attempt to use a cached embedding
        if image_hash in cache:
            # Use the cached embedding
            collection.add(ids=[str(image_path)], embeddings=[cache[image_hash]])
        else:
            # No cached embedding is available, so compute a new one
            # Load the image using PIL
            image = PIL.Image.open(image_path)
            pixel_array = np.array(image)

            # Store the image in the collection
            collection.add(ids=[str(image_path)], images=[pixel_array])

            # Retrieve and store the computed embedding for the image
            embedding = collection.get(ids=[str(image_path)], include=["embeddings"])[
                "embeddings"
            ][0]
            cache[image_hash] = embedding

    # Define the Gradio dashboard render function
    print("Launching Gradio dashboard...")
    with gradio.Blocks() as dashboard:
        input_text = gradio.Textbox(label="Query", placeholder="Picture of a cat")

        @gradio.render(inputs=input_text)
        def search_images(query: str) -> None:
            results = collection.query(query_texts=[query], n_results=10)
            ids = results["ids"][0]
            distances = results["distances"][0]
            for id, distance in zip(ids, distances):
                gradio.Textbox(id, label="Path")
                gradio.Image(id, label="Image")
                gradio.Number(distance, label="Distance")

    # Launch the Gradio dashboard
    dashboard.launch(allowed_paths=[str(directory)], inbrowser=True, share=False)


if __name__ == "__main__":
    main()
