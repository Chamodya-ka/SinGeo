# SinGeo
The official implementation of "SinGeo: Unlock Single Model’s Potential for Robust Cross-View Geo-Localization", accepted at CVPR 2026 as Highlight! You can access the paper [Here](https://arxiv.org/abs/2603.09377)

## Introduction

SinGeo tries to deliver a paradigm shift to learning a single model for robust CVGL via effective combination of proposed dual discriminative learning and curriculum-guided progressive training.

![teaser](teaser.png)

## Environments

Required environments:
- Linux
- Python 3.7+
- PyTorch 1.10.0+
- CUDA 9.2+
- GCC 5+

Please use the following commands to prepare the environment.
```
git clone https://github.com/Yangchen-nudt/SinGeo.git
cd SinGeo
pip install -r requirements.txt
```

## Training:
We provide codes for training on 4 datases (CVUSA/CVACT/VIGOR/University-1652), and take the CVUSA for example.

- Set the path in [dataset](singeo/dataset/cvusa.py), [training](train_singeo_cvusa.py), [distance_calc](calc_distance_cvusa.py).
- Execute the calc_distance_cvusa script:
```
python calc_distance_cvusa.py
``` 
- [Optional] Download the pretrained model weights, and set the model path in [model](singeo/model.py)
- Train the model by running:
```
python train_singeo_cvusa.py
```
- Evaluate the trained model by running:
```
python eval_cvusa.py
```

Note:
- Change the "fov" configuration in the training&evaluating code to change evaluation settings:
```
0.0: north-aligned, value from (70.0, 90.0, 180.0): limited FoV, 360.0: arbitrary orientations
```
- Change the "fov_start"/"fov_end", "min_prob"/"max_prob" in the training code to adjust the curriculum setting.

## Acknowledgement:
We thank the authors of relevant CVGL works for their valuable code bases and benchmarks. If you find SinGeo helpful in your research, please consider citing:

```bibtex
@inproceedings{chen2026singeo,
  title={SinGeo: Unlock Single Model's Potential for Robust Cross-View Geo-Localization},
  author={Chen, Yang and Chen, Xieyuanli and Li, Junxiang and Tang, Jie and Wu, Tao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={19403--19412},
  year={2026}
}

@inproceedings{mi2024congeo,
  title={Congeo: Robust cross-view geo-localization across ground view variations},
  author={Mi, Li and Xu, Chang and Castillo-Navarro, Javiera and Montariol, Syrielle and Yang, Wen and Bosselut, Antoine and Tuia, Devis},
  booktitle={European Conference on Computer Vision},
  pages={214--230},
  year={2024},
  organization={Springer}
}

@inproceedings{deuser2023sample4geo,
  title={Sample4geo: Hard negative sampling for cross-view geo-localisation},
  author={Deuser, Fabian and Habel, Konrad and Oswald, Norbert},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={16847--16856},
  year={2023}
}
```
