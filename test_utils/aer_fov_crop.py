import cv2
import numpy as np


def create_pie_mask(image_shape, center, radius, start_angle, end_angle):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, center, (radius, radius), 0, start_angle, end_angle, 255, -1)
    return mask

def get_oriented_fov_crop(image_path, fov_degrees, rotation_degrees):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found. Check the path.")
    h, w, _ = img.shape
    center_x, center_y = w // 2, h // 2
    radius_px = max(w, h) 
    # mask = create_pie_mask(image_shape=img.shape[:2], center=(center_x, center_y), radius=radius_px, start_angle=(rotation_degrees - fov_degrees//2 -90), end_angle=(rotation_degrees + fov_degrees//2 -90))
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, (center_x, center_y), (radius_px, radius_px), 0, rotation_degrees - fov_degrees//2 -90, rotation_degrees + fov_degrees//2 -90, 255, -1)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    final_crop = masked_img
    return final_crop


x = get_oriented_fov_crop("test_utils/0044020_a.jpg", 180, 90)

cv2.imwrite("test_utils/0044020_a_cropped.jpg", x)