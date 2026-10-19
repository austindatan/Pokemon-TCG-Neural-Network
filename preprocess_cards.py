import os
import sys
import argparse
from pathlib import Path
from typing import Tuple, Optional


try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    from PIL import Image, ImageOps
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

if not HAS_OPENCV and not HAS_PIL:
    raise ImportError("Please install either opencv-python ('pip install opencv-python') or Pillow ('pip install pillow').")

DEFAULT_CROP_RATIOS = {
    "top": 0.125,   
    "bottom": 0.500, 
    "left": 0.105,   
    "right": 0.895    
}


def crop_artwork_cv2(
    image: np.ndarray,
    crop_ratios: dict = DEFAULT_CROP_RATIOS
) -> np.ndarray:
    """
    Crops out the inner artwork illustration box from a Pokémon card using OpenCV.
    
    Parameters:
        image (np.ndarray): Input card image in BGR format (H, W, C).
        crop_ratios (dict): Relative coordinate boundaries ('top', 'bottom', 'left', 'right').
        
    Returns:
        np.ndarray: Cropped artwork sub-image.
    """
    height, width = image.shape[:2]
    
    y_min = int(height * crop_ratios["top"])
    y_max = int(height * crop_ratios["bottom"])
    x_min = int(width * crop_ratios["left"])
    x_max = int(width * crop_ratios["right"])
    
    y_min, y_max = max(0, y_min), min(height, y_max)
    x_min, x_max = max(0, x_min), min(width, x_max)
    
    return image[y_min:y_max, x_min:x_max]


def crop_artwork_pil(
    image: "Image.Image",
    crop_ratios: dict = DEFAULT_CROP_RATIOS
) -> "Image.Image":
    """
    Crops out the inner artwork illustration box from a Pokémon card using Pillow.
    
    Parameters:
        image (PIL.Image): Input card image.
        crop_ratios (dict): Relative coordinate boundaries.
        
    Returns:
        PIL.Image: Cropped artwork sub-image.
    """
    width, height = image.size
    
    x_min = int(width * crop_ratios["left"])
    y_min = int(height * crop_ratios["top"])
    x_max = int(width * crop_ratios["right"])
    y_max = int(height * crop_ratios["bottom"])
    
    return image.crop((x_min, y_min, x_max, y_max))


def resize_and_pad_cv2(
    image: np.ndarray,
    target_size: Tuple[int, int] = (128, 128),
    pad_color: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    Resizes an image while preserving its aspect ratio by applying uniform scaling
    and centering it within a letterbox/pillarbox canvas of the target dimensions.
    
    Parameters:
        image (np.ndarray): Cropped artwork image (H, W, C).
        target_size (tuple): Desired (target_height, target_width), e.g. (128, 128).
        pad_color (tuple): BGR color tuple for border padding (default: black [0, 0, 0]).
        
    Returns:
        np.ndarray: Uniformly resized and padded image of size (target_h, target_w, C).
    """
    target_h, target_w = target_size
    orig_h, orig_w = image.shape[:2]
    
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    
    padded = cv2.copyMakeBorder(
        resized,
        top=pad_top,
        bottom=pad_bottom,
        left=pad_left,
        right=pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_color
    )
    
    return padded


def resize_and_pad_pil(
    image: "Image.Image",
    target_size: Tuple[int, int] = (128, 128),
    pad_color: Tuple[int, int, int] = (0, 0, 0)
) -> "Image.Image":
    """
    Resizes and pads an image using Pillow while preserving aspect ratio.
    """
    target_h, target_w = target_size
    orig_w, orig_h = image.size
    
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    
    resample_filter = getattr(Image, "Resampling", Image).LANCZOS
    resized = image.resize((new_w, new_h), resample=resample_filter)
    
    canvas = Image.new(image.mode, (target_w, target_h), pad_color)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))
    return canvas


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalizes pixel intensities from the standard [0, 255] integer range to [0.0, 1.0] float32.
    
    Why Normalization is Crucial in CNNs (Lab Report Concept):
      - Keeps input features on a consistent scale, preventing vanishing/exploding gradients.
      - Enables faster convergence during gradient descent optimization.
      - Minimizes sensitivity to varying illumination and lighting across card scans.
      
    Parameters:
        image (np.ndarray): 8-bit image array (values 0-255).
        
    Returns:
        np.ndarray: float32 image array with values in [0.0, 1.0].
    """
    return image.astype(np.float32) / 255.0


def process_single_image(
    input_path: Path,
    output_path: Path,
    target_size: Tuple[int, int] = (128, 128),
    crop_ratios: dict = DEFAULT_CROP_RATIOS,
    use_opencv: bool = True
) -> bool:
    """
    Executes the full preprocessing pipeline on a single image file and writes the
    standardized 8-bit image to disk ready for training (with normalization documented).
    """
    try:
        if use_opencv and HAS_OPENCV:
            img = cv2.imread(str(input_path))
            if img is None:
                return False
            
            cropped = crop_artwork_cv2(img, crop_ratios=crop_ratios)
            
            standardized = resize_and_pad_cv2(cropped, target_size=target_size)
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), standardized)
            return True

        elif HAS_PIL:
            with Image.open(input_path) as img:
                img = img.convert("RGB")
                cropped = crop_artwork_pil(img, crop_ratios=crop_ratios)
                standardized = resize_and_pad_pil(cropped, target_size=target_size)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                standardized.save(str(output_path), quality=95)
                return True
        else:
            return False

    except Exception as exc:
        print(f"[Error] Failed to process {input_path.name}: {exc}", file=sys.stderr)
        return False


def batch_preprocess_dataset(
    input_dir: str,
    output_dir: str,
    target_size: Tuple[int, int] = (128, 128),
    crop_ratios: dict = DEFAULT_CROP_RATIOS,
    valid_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".bmp"),
    max_images: Optional[int] = None
) -> dict:
    """
    Recursively scans input_dir, preprocesses every image found, and outputs to
    output_dir matching the exact subdirectory layout.
    """
    in_path = Path(input_dir).resolve()
    out_path = Path(output_dir).resolve()
    
    if not in_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {in_path}")
    
    out_path.mkdir(parents=True, exist_ok=True)
    
    all_files = [
        p for p in in_path.rglob("*")
        if p.is_file() and p.suffix.lower() in valid_extensions
    ]
    
    if max_images is not None and max_images > 0:
        all_files = all_files[:max_images]
        
    total_found = len(all_files)
    print(f"============================================================")
    print(f"Starting Preprocessing Pipeline")
    print(f"  Input Directory  : {in_path}")
    print(f"  Output Directory : {out_path}")
    print(f"  Total Images     : {total_found}")
    print(f"  Target Resolution: {target_size[0]}x{target_size[1]} pixels (Padded)")
    print(f"  Crop Window      : Top={crop_ratios['top']:.1%}, Bottom={crop_ratios['bottom']:.1%}, "
          f"Left={crop_ratios['left']:.1%}, Right={crop_ratios['right']:.1%}")
    print(f"============================================================")
    
    processed_count = 0
    failed_count = 0
    
    for idx, file_p in enumerate(all_files, start=1):
        relative_subpath = file_p.relative_to(in_path)
        dest_p = out_path / relative_subpath
        
        success = process_single_image(
            input_path=file_p,
            output_path=dest_p,
            target_size=target_size,
            crop_ratios=crop_ratios,
            use_opencv=HAS_OPENCV
        )
        
        if success:
            processed_count += 1
        else:
            failed_count += 1
            
        if idx % 100 == 0 or idx == total_found:
            percent = (idx / total_found) * 100
            print(f"  Progress: {idx}/{total_found} ({percent:.1f}%) | Success: {processed_count} | Failed: {failed_count}")
            
    print(f"\n[Completed] Processed {processed_count} images successfully ({failed_count} failed).")
    print(f"Output saved to: {out_path}")
    
    return {
        "total": total_found,
        "processed": processed_count,
        "failed": failed_count,
        "output_dir": str(out_path)
    }


def parse_arguments():
    """Command-line argument parser for flexible execution."""
    parser = argparse.ArgumentParser(
        description="Batch preprocess Pokémon TCG cards: crop artwork box, resize with padding, and normalize."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="2000",
        help="Path to folder containing raw card images (default: '2000')."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="dataset_preprocessed",
        help="Path to output directory (default: 'dataset_preprocessed')."
    )
    parser.add_argument(
        "--size", "-s",
        type=int,
        default=128,
        help="Target square dimension in pixels, e.g. 128 for 128x128 or 224 for 224x224 (default: 128)."
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Optional limit on number of images to process (useful for quick testing)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    batch_preprocess_dataset(
        input_dir=args.input,
        output_dir=args.output,
        target_size=(args.size, args.size),
        max_images=args.limit
    )
