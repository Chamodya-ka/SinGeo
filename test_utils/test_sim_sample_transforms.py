import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from singeo.transforms_fov_aware import get_transforms_val_sim_sample

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "vis_sim_sample")


class TestSimSampleTransforms:
    """Visualization-only checks for get_transforms_val_sim_sample."""

    @staticmethod
    def make_panorama(width=768, height=140):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        for deg in range(0, 360, 30):
            x = int(deg / 360.0 * width)
            color = np.array(plt.cm.hsv(deg / 360.0)[:3]) * 255
            img[:, max(0, x - 2) : x + 2, :] = color.astype(np.uint8)
        return img

    @staticmethod
    def make_aerial(size=384):
        img = np.zeros((size, size, 3), dtype=np.uint8)
        center = (size // 2, size // 2)
        radius = size // 2 - 2

        for deg in range(0, 360, 30):
            color_deg = (deg + 180) % 360
            color = tuple((np.array(plt.cm.hsv(color_deg / 360.0)[:3]) * 255).astype(np.uint8).tolist())
            theta = np.deg2rad(deg)
            dx = np.sin(theta)
            dy = -np.cos(theta)
            end_x = int(center[0] + dx * radius)
            end_y = int(center[1] + dy * radius)
            cv2.line(img, center, (end_x, end_y), color, thickness=3, lineType=cv2.LINE_AA)

        return img

    @staticmethod
    def unnormalize_image(image, mean, std):
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        image = image.astype(np.float32)
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        image = image * np.array(std).reshape(1, 1, 3) + np.array(mean).reshape(1, 1, 3)
        image = np.clip(image, 0.0, 1.0)
        return image

    @staticmethod
    def save_visualization(image, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.imsave(path, image)

    def test_visualize_val_sim_sample(self):
        """Generate and save visualizations for the sim-sample validation transforms."""
        satellite_image = self.make_aerial(size=384)
        ground_image = self.make_panorama(width=768, height=140)

        cases = [
            (360.0, 0.0),
            (90.0, 0.0),
            (90.0, 90.0),
            (180.0, 45.0),
        ]

        for fov, orientation in cases:
            sat_transform, grd_transform = get_transforms_val_sim_sample(
                image_size_sat=(384, 384),
                img_size_ground=(140, 768),
                fov=fov,
                orientation=orientation,
            )

            sat_out = sat_transform(image=satellite_image)["image"]
            grd_out = grd_transform(image=ground_image)["image"]

            sat_vis = self.unnormalize_image(sat_out, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            grd_vis = self.unnormalize_image(grd_out, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            self.save_visualization(sat_vis, os.path.join(OUTPUT_DIR, f"satellite_fov_{int(fov)}_orient_{int(orientation)}.png"))
            self.save_visualization(grd_vis, os.path.join(OUTPUT_DIR, f"ground_fov_{int(fov)}_orient_{int(orientation)}.png"))


if __name__ == "__main__":
    TestSimSampleTransforms().test_visualize_val_sim_sample()
