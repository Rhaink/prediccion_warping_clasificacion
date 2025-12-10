"""
Piecewise Affine Warping for Geometric Normalization.

This module implements piecewise affine warping to transform images
to a canonical shape using predicted landmarks.

Pipeline:
1. Input image -> Model predicts 15 landmarks
2. Predicted landmarks + Canonical shape + Delaunay -> Warping
3. Warped image (geometrically normalized)
"""

import numpy as np
import cv2
import warnings
from typing import Tuple, Optional
from scipy.spatial import Delaunay


def scale_landmarks_from_centroid(
    landmarks: np.ndarray,
    scale: float = 1.0
) -> np.ndarray:
    """
    Scale landmarks from their centroid.

    Args:
        landmarks: Array (n_landmarks, 2) with coordinates
        scale: Scale factor (>1 expands, <1 contracts)

    Returns:
        scaled: Scaled landmarks
    """
    centroid = landmarks.mean(axis=0)
    scaled = centroid + (landmarks - centroid) * scale
    return scaled


def clip_landmarks_to_image(
    landmarks: np.ndarray,
    image_size: int = 224,
    margin: int = 2
) -> np.ndarray:
    """
    Ensure landmarks are within image bounds.

    Args:
        landmarks: Array (n_landmarks, 2) with coordinates
        image_size: Image size
        margin: Minimum distance from image edges

    Returns:
        clipped: Clipped landmarks
    """
    clipped = np.clip(landmarks, margin, image_size - margin - 1)
    return clipped


def add_boundary_points(
    landmarks: np.ndarray,
    image_size: int = 224
) -> np.ndarray:
    """
    Add boundary points to landmarks for full image coverage.

    Adds 8 points: 4 corners + 4 midpoints of edges.

    Args:
        landmarks: Array (15, 2) with original landmarks
        image_size: Image size

    Returns:
        extended_landmarks: Array (23, 2) with landmarks + boundary points
    """
    # 4 corners
    corners = np.array([
        [0, 0],                        # Top-left
        [image_size - 1, 0],           # Top-right
        [0, image_size - 1],           # Bottom-left
        [image_size - 1, image_size - 1]  # Bottom-right
    ], dtype=np.float64)

    # 4 edge midpoints
    midpoints = np.array([
        [image_size / 2, 0],           # Top-center
        [0, image_size / 2],           # Left-center
        [image_size - 1, image_size / 2],  # Right-center
        [image_size / 2, image_size - 1]   # Bottom-center
    ], dtype=np.float64)

    # Concatenate: landmarks + corners + midpoints
    extended = np.vstack([landmarks, corners, midpoints])
    return extended


def get_affine_transform_matrix(
    src_tri: np.ndarray,
    dst_tri: np.ndarray
) -> np.ndarray:
    """
    Compute affine transformation matrix between two triangles.

    Args:
        src_tri: Source triangle vertices (3, 2)
        dst_tri: Destination triangle vertices (3, 2)

    Returns:
        M: 2x3 affine transformation matrix
    """
    src = src_tri.astype(np.float32)
    dst = dst_tri.astype(np.float32)
    M = cv2.getAffineTransform(src, dst)
    return M


def create_triangle_mask(
    shape: Tuple[int, int],
    triangle: np.ndarray
) -> np.ndarray:
    """
    Create binary mask for a triangle.

    Args:
        shape: (height, width) of the image
        triangle: Triangle vertices (3, 2)

    Returns:
        mask: Binary mask with the triangle filled
    """
    mask = np.zeros(shape, dtype=np.uint8)
    pts = triangle.astype(np.int32)
    cv2.fillConvexPoly(mask, pts, 255)
    return mask


def get_bounding_box(triangle: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Get bounding box of a triangle.

    Args:
        triangle: Triangle vertices (3, 2)

    Returns:
        (x, y, w, h) of bounding box
    """
    x_min = max(0, int(np.floor(triangle[:, 0].min())))
    y_min = max(0, int(np.floor(triangle[:, 1].min())))
    x_max = int(np.ceil(triangle[:, 0].max()))
    y_max = int(np.ceil(triangle[:, 1].max()))
    return x_min, y_min, x_max - x_min, y_max - y_min


def warp_triangle(
    src_img: np.ndarray,
    dst_img: np.ndarray,
    src_tri: np.ndarray,
    dst_tri: np.ndarray
) -> None:
    """
    Warp a triangle from source to destination image.

    The warping is IN-PLACE (modifies dst_img directly).

    Args:
        src_img: Source image (H, W) or (H, W, C)
        dst_img: Destination image (must have same size as src_img)
        src_tri: Source triangle vertices (3, 2)
        dst_tri: Destination triangle vertices (3, 2)
    """
    # Get bounding boxes
    src_rect = get_bounding_box(src_tri)
    dst_rect = get_bounding_box(dst_tri)

    # Extract regions of interest
    src_x, src_y, src_w, src_h = src_rect
    dst_x, dst_y, dst_w, dst_h = dst_rect

    # Verify bounding boxes are valid
    if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
        return

    # Adjust triangles to local coordinates
    src_tri_local = src_tri.copy()
    src_tri_local[:, 0] -= src_x
    src_tri_local[:, 1] -= src_y

    dst_tri_local = dst_tri.copy()
    dst_tri_local[:, 0] -= dst_x
    dst_tri_local[:, 1] -= dst_y

    # Extract source patch
    src_patch = src_img[src_y:src_y + src_h, src_x:src_x + src_w].copy()

    # Compute affine transformation
    M = get_affine_transform_matrix(src_tri_local, dst_tri_local)

    # Apply warp to patch
    warped_patch = cv2.warpAffine(
        src_patch, M, (dst_w, dst_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    # Create destination triangle mask
    mask = create_triangle_mask((dst_h, dst_w), dst_tri_local)

    # Apply mask and copy to destination
    if len(dst_img.shape) == 3:
        mask = mask[:, :, np.newaxis]

    # Blend using mask
    dst_region = dst_img[dst_y:dst_y + dst_h, dst_x:dst_x + dst_w]
    np.copyto(dst_region, warped_patch, where=(mask > 0))


def piecewise_affine_warp(
    image: np.ndarray,
    source_landmarks: np.ndarray,
    target_landmarks: np.ndarray,
    triangles: Optional[np.ndarray] = None,
    output_size: int = 224,
    use_full_coverage: bool = True
) -> np.ndarray:
    """
    Piecewise affine warping using triangulation.

    Args:
        image: Source image (H, W) or (H, W, C)
        source_landmarks: Source image landmarks (15, 2)
        target_landmarks: Target/canonical landmarks (15, 2)
        triangles: Triangle indices (n_tri, 3). If None and use_full_coverage=True,
                   extended triangulation is computed.
        output_size: Output image size
        use_full_coverage: If True, add boundary points for full coverage

    Returns:
        warped_image: Image warped to canonical shape
    """
    # If we want full coverage, extend landmarks and recompute triangulation
    if use_full_coverage:
        src_extended = add_boundary_points(source_landmarks, output_size)
        dst_extended = add_boundary_points(target_landmarks, output_size)
        # Compute new triangulation on destination points
        tri = Delaunay(dst_extended)
        triangles = tri.simplices
        source_pts = src_extended
        target_pts = dst_extended
    else:
        source_pts = source_landmarks
        target_pts = target_landmarks

    # Create destination image
    if len(image.shape) == 3:
        warped = np.zeros((output_size, output_size, image.shape[2]), dtype=image.dtype)
    else:
        warped = np.zeros((output_size, output_size), dtype=image.dtype)

    # Warp each triangle
    for tri_indices in triangles:
        src_tri = source_pts[tri_indices]
        dst_tri = target_pts[tri_indices]

        # Verify triangles are valid (not degenerate)
        # Use cross product formula for 2D triangle area: 0.5 * |v1 x v2|
        # where v1 = tri[1] - tri[0], v2 = tri[2] - tri[0]
        def triangle_area_2x(tri):
            v1 = tri[1] - tri[0]
            v2 = tri[2] - tri[0]
            return abs(v1[0] * v2[1] - v1[1] * v2[0])

        src_area = triangle_area_2x(src_tri)
        dst_area = triangle_area_2x(dst_tri)

        if src_area < 1e-6 or dst_area < 1e-6:
            continue

        try:
            warp_triangle(image, warped, src_tri, dst_tri)
        except Exception as e:
            warnings.warn(f"Error warping triangle {tri_indices}: {e}")
            continue

    return warped


def compute_fill_rate(warped_image: np.ndarray) -> float:
    """
    Compute fill rate of a warped image.

    Fill rate is the proportion of non-black pixels.

    Args:
        warped_image: Warped image

    Returns:
        fill_rate: Value between 0 and 1
    """
    black_pixels = np.sum(warped_image == 0)
    fill_rate = 1 - (black_pixels / warped_image.size)
    return fill_rate
