"""
DeFungi Dataset Preparation Script
===================================
1. Downloads the DeFungi dataset from UCI ML Repository
2. Extracts the images
3. Downscales all images to 64x64 pixels
4. Splits into Training (70%), Validation (15%), Testing (15%)
5. Organizes into structured directories
"""

import os
import sys
import zipfile
import shutil
import random
import time
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Pillow not installed. Installing...")
    os.system(f"{sys.executable} -m pip install Pillow")
    from PIL import Image

try:
    import requests
except ImportError:
    print("requests not installed. Installing...")
    os.system(f"{sys.executable} -m pip install requests")
    import requests

# ============================================================
# Configuration
# ============================================================
BASE_DIR = Path(r"d:\Sem 5\Pattern Recognition\CNN Assignment")
DOWNLOAD_URL = "https://archive.ics.uci.edu/static/public/773/defungi.zip"
ZIP_PATH = BASE_DIR / "defungi.zip"
EXTRACT_DIR = BASE_DIR / "defungi_raw"
OUTPUT_DIR = BASE_DIR / "dataset"
IMG_SIZE = (64, 64)  # Target resolution
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42

# ============================================================
# Step 1: Download the dataset (with retries)
# ============================================================
def download_dataset():
    if ZIP_PATH.exists() and ZIP_PATH.stat().st_size > 1_000_000:
        print(f"[INFO] ZIP file already exists at {ZIP_PATH}. Skipping download.")
        return
    
    # Remove partial downloads
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    
    print(f"[INFO] Downloading DeFungi dataset from {DOWNLOAD_URL}...")
    print("[INFO] This may take a few minutes (~149 MB)...")
    
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[INFO] Download attempt {attempt}/{max_retries}...")
            response = requests.get(DOWNLOAD_URL, stream=True, timeout=120)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            chunk_size = 8192 * 16  # 128KB chunks
            
            with open(str(ZIP_PATH), 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = int(downloaded * 100 / total_size)
                            sys.stdout.write(f"\r[DOWNLOAD] {percent}% ({downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB)")
                        else:
                            sys.stdout.write(f"\r[DOWNLOAD] {downloaded // (1024*1024)}MB downloaded")
                        sys.stdout.flush()
            
            print(f"\n[INFO] Download complete! File size: {ZIP_PATH.stat().st_size / (1024*1024):.1f} MB")
            return  # Success
            
        except Exception as e:
            print(f"\n[WARNING] Download attempt {attempt} failed: {e}")
            if ZIP_PATH.exists():
                ZIP_PATH.unlink()
            if attempt < max_retries:
                wait = attempt * 5
                print(f"[INFO] Retrying in {wait} seconds...")
                time.sleep(wait)
            else:
                print("[ERROR] All download attempts failed.")
                print("[TIP] You can manually download from:")
                print(f"  {DOWNLOAD_URL}")
                print(f"  and save it to: {ZIP_PATH}")
                sys.exit(1)

# ============================================================
# Step 2: Extract the dataset
# ============================================================
def extract_dataset():
    if EXTRACT_DIR.exists() and any(EXTRACT_DIR.iterdir()):
        print(f"[INFO] Extracted data already exists at {EXTRACT_DIR}. Skipping extraction.")
        return
    print(f"[INFO] Extracting dataset to {EXTRACT_DIR}...")
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(ZIP_PATH), 'r') as zip_ref:
        zip_ref.extractall(str(EXTRACT_DIR))
    print("[INFO] Extraction complete!")

# ============================================================
# Step 3: Discover classes and images
# ============================================================
def discover_images():
    """Find all image files organized by class (folder name)."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    class_images = {}
    
    # Walk through extracted directory to find class folders
    for root, dirs, files in os.walk(str(EXTRACT_DIR)):
        for f in files:
            if Path(f).suffix.lower() in image_extensions:
                # The class is determined by the immediate parent folder name
                class_name = Path(root).name
                if class_name not in class_images:
                    class_images[class_name] = []
                class_images[class_name].append(os.path.join(root, f))
    
    return class_images

# ============================================================
# Step 4: Split and save with downscaling
# ============================================================
def split_and_save(class_images):
    """Split images into train/val/test and save resized copies."""
    random.seed(RANDOM_SEED)
    
    # Create output directories
    for split in SPLIT_RATIOS.keys():
        for class_name in class_images.keys():
            (OUTPUT_DIR / split / class_name).mkdir(parents=True, exist_ok=True)
    
    total_stats = {"train": 0, "val": 0, "test": 0}
    class_stats = {}
    
    print(f"\n{'='*60}")
    print(f"{'CLASS':<15} {'TOTAL':<10} {'TRAIN':<10} {'VAL':<10} {'TEST':<10}")
    print(f"{'='*60}")
    
    for class_name, images in sorted(class_images.items()):
        # Shuffle images
        random.shuffle(images)
        n = len(images)
        
        # Calculate split indices
        n_train = int(n * SPLIT_RATIOS["train"])
        n_val = int(n * SPLIT_RATIOS["val"])
        # Remaining goes to test
        n_test = n - n_train - n_val
        
        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:]
        }
        
        class_stats[class_name] = {"total": n}
        
        for split_name, split_images in splits.items():
            for img_path in split_images:
                try:
                    img = Image.open(img_path).convert("RGB")
                    img_resized = img.resize(IMG_SIZE, Image.LANCZOS)
                    
                    # Save with original filename
                    filename = Path(img_path).name
                    save_path = OUTPUT_DIR / split_name / class_name / filename
                    img_resized.save(str(save_path))
                except Exception as e:
                    print(f"[WARNING] Failed to process {img_path}: {e}")
            
            total_stats[split_name] += len(split_images)
            class_stats[class_name][split_name] = len(split_images)
        
        print(f"{class_name:<15} {n:<10} {class_stats[class_name]['train']:<10} "
              f"{class_stats[class_name]['val']:<10} {class_stats[class_name]['test']:<10}")
    
    print(f"{'='*60}")
    total = sum(total_stats.values())
    print(f"{'TOTAL':<15} {total:<10} {total_stats['train']:<10} "
          f"{total_stats['val']:<10} {total_stats['test']:<10}")
    print(f"{'='*60}")
    
    # Print percentages
    print(f"\n[INFO] Split Percentages:")
    print(f"  Train: {total_stats['train']/total*100:.1f}%")
    print(f"  Val:   {total_stats['val']/total*100:.1f}%")
    print(f"  Test:  {total_stats['test']/total*100:.1f}%")
    
    return class_stats, total_stats

def extract_processed_zip():
    """Extract dataset_64x64.zip directly if available."""
    zip_path = BASE_DIR / "dataset_64x64.zip"
    if zip_path.exists():
        print(f"[INFO] Found pre-packaged 64x64 dataset: {zip_path}")
        print("[INFO] Extracting directly (takes ~2 seconds)...")
        with zipfile.ZipFile(str(zip_path), 'r') as zf:
            zf.extractall(str(BASE_DIR))
        print("[INFO] Extracted successfully to dataset/ directory!")
        return True
    return False

# ============================================================
# Main Execution
# ============================================================
def main():
    print("=" * 60)
    print("DeFungi Dataset Preparation")
    print("=" * 60)
    print(f"Target resolution: {IMG_SIZE[0]}x{IMG_SIZE[1]} pixels")
    print(f"Split: Train={SPLIT_RATIOS['train']*100:.0f}% | "
          f"Val={SPLIT_RATIOS['val']*100:.0f}% | "
          f"Test={SPLIT_RATIOS['test']*100:.0f}%")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)

    # Check if dataset directory already exists
    if OUTPUT_DIR.exists() and (OUTPUT_DIR / "train").exists():
        n_train = sum(1 for _ in (OUTPUT_DIR / "train").rglob("*.jpg"))
        n_val = sum(1 for _ in (OUTPUT_DIR / "val").rglob("*.jpg"))
        n_test = sum(1 for _ in (OUTPUT_DIR / "test").rglob("*.jpg"))
        if n_train > 0 and n_val > 0 and n_test > 0:
            print(f"[INFO] Dataset already prepared and verified!")
            print(f"  Train: {n_train} | Val: {n_val} | Test: {n_test} | Total: {n_train+n_val+n_test}")
            return

    # Option 1: Fast extract from pre-packaged 64x64 zip (10MB)
    if extract_processed_zip():
        return

    # Option 2: Full download and processing from UCI ML Repository
    # Step 1: Download
    download_dataset()
    
    # Step 2: Extract
    extract_dataset()
    
    # Step 3: Discover images
    print("\n[INFO] Discovering images and classes...")
    class_images = discover_images()
    
    if not class_images:
        print("[ERROR] No images found! Check the extraction directory.")
        sys.exit(1)
    
    print(f"[INFO] Found {len(class_images)} classes:")
    for cls, imgs in sorted(class_images.items()):
        print(f"  - {cls}: {len(imgs)} images")
    
    total_images = sum(len(v) for v in class_images.values())
    print(f"[INFO] Total images: {total_images}")
    
    # Step 4: Split and save resized images
    print(f"\n[INFO] Resizing to {IMG_SIZE[0]}x{IMG_SIZE[1]} and splitting...")
    class_stats, total_stats = split_and_save(class_images)
    
    print(f"\n[INFO] Dataset preparation complete!")
    print(f"[INFO] Output structure:")
    print(f"  {OUTPUT_DIR}/")
    print(f"    |-- train/     ({total_stats['train']} images)")
    print(f"    |-- val/       ({total_stats['val']} images)")
    print(f"    +-- test/      ({total_stats['test']} images)")
    
    # Summary for assignment sheet
    print(f"\n{'='*60}")
    print(f"DATASET SUMMARY (for assignment sheet)")
    print(f"{'='*60}")
    print(f"Dataset Name: DeFungi")
    print(f"Source: https://archive.ics.uci.edu/dataset/773/defungi")
    print(f"Description: Microscopic fungi images for classification")
    print(f"             (superficial fungal infections - yeasts, moulds, dermatophytes)")
    print(f"Number of Classes: {len(class_images)}")
    print(f"Total Images: {total_images}")
    print(f"Image Resolution: {IMG_SIZE[0]}x{IMG_SIZE[1]} pixels (downscaled)")
    print(f"Training Set: {total_stats['train']} images ({total_stats['train']/total_images*100:.1f}%)")
    print(f"Validation Set: {total_stats['val']} images ({total_stats['val']/total_images*100:.1f}%)")
    print(f"Test Set: {total_stats['test']} images ({total_stats['test']/total_images*100:.1f}%)")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
