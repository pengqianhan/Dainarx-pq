#!/usr/bin/env python3
"""
Image Comparison Script

Compares two PNG images using:
1. Binary/byte-level comparison for exact match checking
2. Visual diff analysis with quantitative metrics when images differ
"""

import sys
import os
import filecmp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def compare_images(img1_path, img2_path, output_dir="result"):
    """
    Compare two images and generate visual diff if they differ.
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        output_dir: Directory to save diff output
    
    Returns:
        bool: True if images are identical, False otherwise
    """
    # Check if files exist
    if not os.path.exists(img1_path):
        print(f"Error: File not found: {img1_path}")
        return None
    if not os.path.exists(img2_path):
        print(f"Error: File not found: {img2_path}")
        return None
    
    print(f"Comparing:")
    print(f"  Image 1: {img1_path}")
    print(f"  Image 2: {img2_path}")
    print()
    
    # Step 1: Binary comparison
    print("Step 1: Binary (byte-level) comparison...")
    are_identical = filecmp.cmp(img1_path, img2_path, shallow=False)
    
    if are_identical:
        print("✓ Images are BYTE-IDENTICAL")
        print("  The two PNG files are exactly the same at the binary level.")
        return True
    
    print("✗ Images are NOT byte-identical")
    print("  Proceeding with visual difference analysis...")
    print()
    
    # Step 2: Load images for visual comparison
    try:
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
    except Exception as e:
        print(f"Error loading images: {e}")
        return False
    
    # Check dimensions
    if img1.size != img2.size:
        print(f"✗ Images have different dimensions:")
        print(f"  Image 1: {img1.size}")
        print(f"  Image 2: {img2.size}")
        print("  Cannot perform pixel-wise comparison.")
        return False
    
    print(f"Image dimensions: {img1.size[0]} x {img1.size[1]}")
    
    # Convert to numpy arrays
    arr1 = np.array(img1, dtype=np.float32)
    arr2 = np.array(img2, dtype=np.float32)
    
    # Step 3: Compute difference metrics
    print("\nStep 2: Visual difference metrics...")
    
    # Absolute difference
    diff = np.abs(arr1 - arr2)
    
    # Mean Squared Error
    mse = np.mean((arr1 - arr2) ** 2)
    
    # Maximum difference
    max_diff = np.max(diff)
    
    # Percentage of different pixels (threshold > 0)
    different_pixels = np.any(diff > 0, axis=2)
    percent_diff = (np.sum(different_pixels) / different_pixels.size) * 100
    
    # Average difference magnitude (for pixels that differ)
    if np.sum(different_pixels) > 0:
        avg_diff_magnitude = np.mean(diff[different_pixels])
    else:
        avg_diff_magnitude = 0
    
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print(f"  Max pixel difference: {max_diff:.2f} (out of 255)")
    print(f"  Pixels that differ: {percent_diff:.2f}%")
    print(f"  Average difference magnitude: {avg_diff_magnitude:.2f}")
    
    # Step 4: Generate visual diff
    print("\nStep 3: Generating visual difference output...")
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare diff visualization
    # Convert diff to grayscale intensity (max across RGB channels)
    diff_intensity = np.max(diff, axis=2)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Image 1
    axes[0, 0].imshow(arr1.astype(np.uint8))
    axes[0, 0].set_title('Image 1: data_duffing/sample_0.png', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Image 2
    axes[0, 1].imshow(arr2.astype(np.uint8))
    axes[0, 1].set_title('Image 2: data_duffing_simulation/sample_0.png', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Difference heatmap
    im = axes[1, 0].imshow(diff_intensity, cmap='hot', vmin=0, vmax=255)
    axes[1, 0].set_title('Difference Heatmap (Pixel-wise Absolute Difference)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04, label='Difference Magnitude')
    
    # Difference overlay on Image 1
    # Highlight different pixels in red overlay
    overlay = arr1.copy().astype(np.uint8)
    diff_mask = different_pixels
    overlay[diff_mask] = [255, 0, 0]  # Red for differences
    
    # Blend original with overlay
    alpha = 0.5
    blended = (alpha * arr1 + (1 - alpha) * overlay).astype(np.uint8)
    
    axes[1, 1].imshow(blended)
    axes[1, 1].set_title('Difference Overlay (Red = Different Pixels)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Add text box with metrics
    metrics_text = (
        f"Comparison Metrics:\n"
        f"MSE: {mse:.4f}\n"
        f"Max Diff: {max_diff:.2f}/255\n"
        f"Different Pixels: {percent_diff:.2f}%\n"
        f"Avg Diff Magnitude: {avg_diff_magnitude:.2f}"
    )
    
    # Add legend for overlay
    red_patch = mpatches.Patch(color='red', alpha=0.5, label='Different pixels')
    axes[1, 1].legend(handles=[red_patch], loc='upper right', fontsize=10)
    
    # Add overall title
    fig.suptitle('Image Comparison: Pixel-by-Pixel Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Add metrics as text
    fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=11, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    # Save output
    output_path = os.path.join(output_dir, 'image_comparison_sample_0.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visual diff saved to: {output_path}")
    
    plt.close()
    
    return False


def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_images.py <image1_path> <image2_path> [output_dir]")
        print("\nExample:")
        print("  python compare_images.py data_duffing/sample_0.png data_duffing_simulation/sample_0.png")
        sys.exit(1)
    
    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "result"
    
    print("=" * 70)
    print("IMAGE COMPARISON TOOL")
    print("=" * 70)
    print()
    
    result = compare_images(img1_path, img2_path, output_dir)
    
    print()
    print("=" * 70)
    if result is True:
        print("CONCLUSION: Images are IDENTICAL (byte-level match)")
    elif result is False:
        print("CONCLUSION: Images are DIFFERENT (see visual diff output)")
    else:
        print("CONCLUSION: Comparison failed")
    print("=" * 70)


if __name__ == "__main__":
    main()

