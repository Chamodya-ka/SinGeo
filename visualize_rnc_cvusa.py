"""Dump RNC crop samples + overlap scores for the first, middle and last epoch.

Standalone inspection script -- it touches no model and no training state, it
just replays the curriculum the training script would apply and renders what the
dataset produces at each stage.

    python visualize_rnc_cvusa.py

Writes one PNG per captured epoch into `output_dir` and prints the overlap
scores. Use it to sanity-check that the ground FoV narrows, the aerial sector
narrows and rotates further, and that the overlap labels fall accordingly.
"""

from dataclasses import dataclass

from singeo.dataset.cvusa import CVUSADatasetTrainSinGeo
from singeo.transforms import (build_satellite_dynamic_transforms, get_dynamic_fov,
                               get_dynamic_rotation_angle, get_transforms_train_singeo,
                               get_transforms_train_singeo_rot)
from singeo.visualize import RnCSampleVisualizer


@dataclass
class Configuration:
    # Mirror the values in train_singeo_cvusa.py so the visualization matches
    # what training would actually show the model.
    data_folder: str = "/home/71/25021871/data/data/cvusa/CVPR_subset"
    output_dir: str = "./rnc_viz"

    img_size: int = 384
    epochs: int = 40
    batch_size: int = 16

    num_samples: int = 5

    # Curriculum, same as the training config.
    fov_start: float = 360.0
    fov_end: float = 70.0

    enable_aerial_crop: bool = True
    aerial_arc_start: float = 360.0
    aerial_arc_end: float = 120.0
    aerial_rot_max: float = 180.0

    # Dropout severity on the cropped views; 1.0 = original SinGeo behaviour.
    crop_dropout_strength: float = 0.5

    rnc_positive_scale: float = 0.5
    # Keep in step with train_singeo_cvusa.py's rnc_positive_overlap, or the
    # figures show a different ranking than the one being optimised.
    rnc_positive_overlap: str = "circle"

    prob_rotate: float = 0.75
    prob_flip: float = 0.5

    # ConvNeXt/timm defaults; only used to undo normalization for display.
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)


config = Configuration()


if __name__ == '__main__':

    image_size_sat = (config.img_size, config.img_size)
    new_width = config.img_size * 2
    new_hight = round((224 / 1232) * new_width)
    img_size_ground = (new_hight, new_width)

    mean, std = list(config.mean), list(config.std)

    sat_t1, sat_t2, ground_t1, ground_t2 = get_transforms_train_singeo(
        image_size_sat, img_size_ground, mean=mean, std=std)

    dataset = CVUSADatasetTrainSinGeo(data_folder=config.data_folder,
                                      transforms_query1=ground_t1,
                                      transforms_query2=ground_t2,
                                      transforms_reference1=sat_t1,
                                      transforms_reference2=sat_t2,
                                      prob_flip=config.prob_flip,
                                      prob_rotate=config.prob_rotate,
                                      shuffle_batch_size=config.batch_size,
                                      return_meta=True,
                                      enable_aerial_crop=config.enable_aerial_crop)

    visualizer = RnCSampleVisualizer.for_schedule(config.output_dir,
                                                  mean, std,
                                                  total_epochs=config.epochs,
                                                  num_samples=config.num_samples,
                                                  positive_scale=config.rnc_positive_scale,
                                                  positive_overlap=config.rnc_positive_overlap)

    print("Capturing epochs:", visualizer.epochs)

    for epoch in visualizer.epochs:

        # --- replay the per-epoch curriculum from train_singeo_cvusa.py ---

        # Aerial view 2 keeps its photometric augs; the discrete +-90 rotation
        # is off because the sector crop applies its own recorded rotation.
        rotate_prob = 1.0 if config.enable_aerial_crop else get_dynamic_fov(
            epoch, config.epochs, fov_start=1.0, fov_end=0.25)
        dataset.transforms_reference2 = build_satellite_dynamic_transforms(
            image_size_sat, mean, std, rotate_prob,
            dropout_strength=config.crop_dropout_strength)

        # Ground view 2: transform built with fov=0 because the dataset does
        # the crop itself and records the window.
        _, _, _, ground_transforms_dynamic = get_transforms_train_singeo_rot(
            image_size_sat, img_size_ground, mean=mean, std=std, fov=0.0,
            con_dropout_strength=config.crop_dropout_strength)
        dataset.transforms_query2 = ground_transforms_dynamic

        dataset.ground_fov = get_dynamic_fov(epoch, config.epochs,
                                             fov_start=config.fov_start,
                                             fov_end=config.fov_end)

        if config.enable_aerial_crop:
            dataset.sat_arc = get_dynamic_fov(epoch, config.epochs,
                                              fov_start=config.aerial_arc_start,
                                              fov_end=config.aerial_arc_end)
            dataset.sat_rot_max = get_dynamic_rotation_angle(epoch, config.epochs,
                                                             min_angle=0.0,
                                                             max_angle=config.aerial_rot_max)

        print("\n{}[Epoch: {}]{}".format(20 * "-", epoch, 20 * "-"))
        print("Ground FoV = {:.2f}, aerial sector = {:.2f}, max tile rotation = {:.2f}".format(
            dataset.ground_fov, dataset.sat_arc, dataset.sat_rot_max))

        visualizer.capture(dataset, epoch)

    print("\nDone. Figures written to:", config.output_dir)
