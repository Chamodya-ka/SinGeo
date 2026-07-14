from singeo.transforms import LimitedFoVCropGrdAerPair
import cv2
from singeo.utils import LabelGenerator


transform = LimitedFoVCropGrdAerPair(fov=90, aerial_fov=90, grd_orientation_shift=0, aer_orientation_shift=0, pad=True, pad_mean=[123.7, 116.3, 103.5])
# test transform on aerial and ground image pair
aer_img = cv2.imread("test_utils/0024271_a.jpg")
grd_img = cv2.imread("test_utils/0024271_g.jpg")

# convert to tensor-like format (H, W, C) -> (C, H, W)
aer_img = aer_img.transpose(2, 0, 1)
grd_img = grd_img.transpose(2, 0, 1)

grd_fov = 90
aerial_fov = 180
grd_orientation_shift = 270
aer_orientation_shift=180

transformed_grd_img, transformed_aer_img = transform(
    image2=aer_img, image1=grd_img,
    fov=grd_fov, aerial_fov=aerial_fov, grd_orientation_shift=grd_orientation_shift, aer_orientation_shift=aer_orientation_shift, pad=True, pad_mean=[123.7, 116.3, 103.5])
print(f"transformed_aer_img.shape: {transformed_aer_img.shape}, transformed_grd_img.shape: {transformed_grd_img.shape}")

# (C, H, W) -> (H, W, C) for both, before writing with cv2
# transformed_aer_img = transformed_aer_img.transpose(1, 2, 0)
transformed_grd_img = transformed_grd_img.transpose(1, 2, 0)

cv2.imwrite("test_utils/transformed_0024271_a_cropped.jpg", transformed_aer_img)
cv2.imwrite("test_utils/transformed_0024271_g_cropped.jpg", transformed_grd_img)


# test overlapping label

print(f"Label: {LabelGenerator(aerial_fov=aerial_fov, grd_fov=grd_fov, grd_orientation_shift=grd_orientation_shift, aerial_orientation_shift=aer_orientation_shift)}")