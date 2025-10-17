import os
import pandas as pd
import sys
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from pathlib import Path

# --- Configuration ---
# Define the paths to your data and where to save images.
DATA_DIR = "dataset/"
IMAGE_DIR = "images/"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
# ---------------------

# --- Worker function for parallel downloading ---
def download_image(task):
    """
    Downloads a single image from a URL and saves it using the sample_id.
    Returns the new filename on success, None on failure.
    """
    link, sample_id = task
    try:
        # Get the original filename to extract its extension (e.g., .jpg)
        original_filename = link.split('/')[-1].split('?')[0]
        _, extension = os.path.splitext(original_filename)

        # If no extension is found in the URL, default to .jpg
        if not extension:
            extension = ".jpg"

        # Create the new filename using the sample_id and the extracted extension
        filename = f"{sample_id}{extension}"
        save_path = os.path.join(IMAGE_DIR, filename)

        # Skip downloading if the file already exists
        if os.path.exists(save_path):
            return None # Return None to indicate it was skipped, not newly downloaded

        # Make the request to get the image data
        response = requests.get(link, stream=True, timeout=15)
        response.raise_for_status()

        # Save the image to the file
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Return the new filename on success
        return filename

    except requests.exceptions.RequestException as e:
        # Print the error for the specific failed link
        print(f"-> ERROR for {link}. Reason: {e}")
        return None # Return None on failure
    except Exception as e:
        print(f"-> UNEXPECTED ERROR for {link}. Reason: {e}")
        return None # Return None on failure


def main():
    """
    Main function to read the CSV, download all training images in parallel,
    and provide a final summary of downloaded files.
    """
    print("--- Starting Image Downloader for train.csv ---")

    # Check if the training CSV file exists
    if not os.path.exists(TRAIN_CSV):
        print(f"FATAL ERROR: Training file not found at '{TRAIN_CSV}'")
        return

    # Read the training data
    try:
        train_df = pd.read_csv(TRAIN_CSV)
        print(f"Successfully loaded {TRAIN_CSV} with {len(train_df)} records.")
    except Exception as e:
        print(f"FATAL ERROR: Could not read the CSV file. Reason: {e}")
        return

    # Create a list of tasks, where each task is a tuple of (link, sample_id)
    # This ensures we keep the link and its corresponding ID together
    tasks_df = train_df[['image_link', 'sample_id']].dropna().drop_duplicates(subset=['image_link'])
    download_tasks = list(tasks_df.to_records(index=False))
    print(f"Found {len(download_tasks)} unique image links to check.")

    # --- PARALLEL DOWNLOAD LOGIC ---
    if download_tasks:
        # Determine the number of workers based on available CPUs, capped at 16
        num_workers = min(64, os.cpu_count() or 1)
        print(f"\nUsing {num_workers} CPU cores to download new images to '{IMAGE_DIR}'...")
        os.makedirs(IMAGE_DIR, exist_ok=True)

        successfully_downloaded = []
        # Use ThreadPoolExecutor to download images in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # tqdm creates a progress bar for the parallel execution
            results = list(tqdm(executor.map(download_image, download_tasks), total=len(download_tasks), desc="Downloading"))

        # Filter out None values from results (which are skipped or failed downloads)
        successfully_downloaded = [res for res in results if res is not None]

    # --- FINAL SUMMARY ---
    print("\n--- Download Process Complete! ---")

    if successfully_downloaded:
        print(f"\nTotal new images successfully downloaded: {len(successfully_downloaded)}")
        print("\nList of newly downloaded images:")
        for name in successfully_downloaded:
            print(f"- {name}")
    else:
        print("\nNo new images were downloaded. All images may already exist or all links failed.")


if __name__ == "__main__":
    main()
