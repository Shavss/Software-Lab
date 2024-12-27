"""
@file post_processing.py
@brief Module for post-processing model predictions.

This module provides functions to perform skeletonization, line detection using Hough Transform,
merging of nearby lines, and saving refined masks as SVG files.
"""

import numpy as np
import cv2
from skimage.morphology import skeletonize, binary_dilation, disk
from skimage.morphology import remove_small_objects, binary_closing
import svgwrite

def post_process_mask(mask):
    """
    @brief Performs skeletonization and morphological cleaning on a predicted mask.

    This function skeletonizes the binary mask to reduce lines to single-pixel width,
    then applies dilation to connect fragmented lines.

    @param mask (numpy.ndarray): Predicted binary mask with shape (160, 160, 1).

    @return numpy.ndarray: Post-processed binary mask.
    """
    mask = mask.squeeze()
    mask = mask.astype(np.uint8)
    # Skeletonize the mask
    skeleton = skeletonize(mask).astype(np.uint8)
    # Apply dilation to connect fragmented lines
    skeleton = binary_dilation(skeleton, disk(1)).astype(np.uint8)
    return skeleton

def detect_lines_skeleton(skeleton):
    """
    @brief Detects lines in a skeletonized mask using Probabilistic Hough Transform.

    This function applies the Hough Line Transform to detect lines within the skeleton.

    @param skeleton (numpy.ndarray): Skeletonized binary mask.

    @return numpy.ndarray or None: Detected lines as an array of endpoints or None if no lines detected.
    """
    lines = cv2.HoughLinesP(
        skeleton,
        rho=1,
        theta=np.pi / 180,
        threshold=30,        # Slightly increased for better precision
        minLineLength=2,
        maxLineGap=3         # Reduced to avoid over-connecting
    )
    return lines

def merge_nearby_lines(lines, proximity_threshold=10, angle_threshold=5):
    """
    @brief Merges duplicate or nearby lines based on proximity and angle thresholds.

    This function iterates through detected lines and merges lines that are close to each other
    and have similar orientations.

    @param lines (numpy.ndarray or None): Detected lines from Hough Transform.
    @param proximity_threshold (int, optional): Maximum distance between lines to consider merging. Default is 10.
    @param angle_threshold (float, optional): Maximum angle difference in degrees to consider merging. Default is 5.

    @return numpy.ndarray or None: Merged lines or None if no lines to merge.
    """
    if lines is None:
        return None

    merged_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        merged = False
        for m_line in merged_lines:
            mx1, my1, mx2, my2 = m_line[0]

            # Calculate proximity and angle
            distance = np.sqrt((x1 - mx1)**2 + (y1 - my1)**2)
            angle_diff = np.abs(
                np.arctan2(y2 - y1, x2 - x1) - np.arctan2(my2 - my1, mx2 - mx1)
            ) * 180 / np.pi

            if distance < proximity_threshold and angle_diff < angle_threshold:
                # Merge lines by averaging their endpoints
                new_x1 = int((x1 + mx1) / 2)
                new_y1 = int((y1 + my1) / 2)
                new_x2 = int((x2 + mx2) / 2)
                new_y2 = int((y2 + my2) / 2)
                m_line[0] = [new_x1, new_y1, new_x2, new_y2]
                merged = True
                break
        if not merged:
            merged_lines.append(line)

    return np.array(merged_lines)

def refine_and_save_as_svg(pred_masks, svg_output_dir='vectorized_svgs_final_please'):
    """
    @brief Refines predicted masks and saves them as SVG files.

    This function post-processes predicted masks, detects lines, merges nearby lines,
    and saves the refined lines as SVG files.

    @param pred_masks (numpy.ndarray): Predicted masks with shape (num_samples, 160, 160, 1).
    @param svg_output_dir (str, optional): Directory to save the SVG files. Default is 'vectorized_svgs_final_please'.

    @return None
    """
    if not os.path.exists(svg_output_dir):
        os.makedirs(svg_output_dir)

    for i, mask in enumerate(pred_masks):
        skeleton = post_process_mask(mask)
        lines = detect_lines_skeleton(skeleton)
        filtered_lines = merge_nearby_lines(lines)

        if filtered_lines is not None:
            # Save as SVG
            svg_filename = f"sample_{i:03d}.svg"
            svg_path = os.path.join(svg_output_dir, svg_filename)
            height, width = skeleton.shape
            dwg = svgwrite.Drawing(svg_path, size=(width, height))
            for line in filtered_lines:
                for x1, y1, x2, y2 in line:
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    dwg.add(dwg.line(
                        start=(x1, y1),
                        end=(x2, y2),
                        stroke="rgb(10,10,16)",
                        stroke_width=2
                    ))
            dwg.save()
        else:
            print(f"No lines detected for sample {i}")

def debug_detected_lines(skeleton, lines):
    """
    @brief Visualizes detected lines on a skeletonized mask for debugging purposes.

    This function plots the skeleton image and overlays the detected lines.

    @param skeleton (numpy.ndarray): Skeletonized binary mask.
    @param lines (numpy.ndarray or None): Detected lines from Hough Transform.

    @return None
    """
    plt.figure()
    plt.imshow(skeleton, cmap="gray")
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                plt.plot([x1, x2], [y1, y2], 'r-', linewidth=2)
    plt.title("Detected Lines on Skeleton")
    plt.axis("off")
    plt.show()
