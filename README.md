# 🌀 PhysFlow

### <p align="left">[Project Page](https://zhuomanliu.github.io/PhysFlow/) | [ArXiv](https://arxiv.org/abs/2411.14423) | [Data](https://drive.google.com/drive/folders/14hrwGOe1MRSySiIa5HK7_SMeNebNdevc?usp=sharing)</p>
####  <p align="left"> [Zhuoman Liu](https://zhuomanliu.tech/), [Weicai Ye](https://ywcmaike.github.io/), [Yan Luximon](), [Pengfei Wan](), [Di Zhang]()</p>

Official implementation of 🌀 **PhysFlow**: Unleashing the Potential of Multi-modal Foundation Models and Video Diffusion for 4D Dynamic Physical Scene Simulation (**CVPR 2025**)


<p align="left">
  <img width="60%" src="assets/teaser.png"/>
  <img width="30%" src="assets/teaser.gif"/>
</p>


## 1. Installation
```sh
conda create -n physflow python=3.9 -y
conda activate physflow

pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
conda env update --file environment.yml

cd submodules
pip install ./simple-knn
pip install ./diff-plane-rasterization
```


## 2. Dataset
We provide three preprocessed datasets: [synthetic dataset from PAC-NeRF](https://drive.google.com/drive/folders/1h79csXUK4clx5Udmsje-9vTwuQ3rBlel?usp=sharing), [real-world dataset from PhysDreamer](https://drive.google.com/drive/folders/1JWXweRK6e1hK7jJzGSJFwSSqNDjLPumo?usp=sharing), and [other real-world scenes](https://drive.google.com/drive/folders/1n9m9AAOxHDiOPPHRrCs0eWRHXdmJPY_u?usp=sharing). Please download them and place in the `./model` directory.

After downloading, the dataset structure will be as follows:
```
model/
├── pacnerf/
│   ├── bird/
│   │   ├── images
│   │   ├── images_generated
│   │   ├── point_cloud
│   │   ├── pcd
│   │   ├── ...
│   │   └── cameras.json
│   │   ...
├── phys_dreamer/
│   ├── alocasia/
│   │   ├── images
│   │   ├── images_generated
│   │   ├── point_cloud
│   │   ├── ...
│   │   └── moving_part_points.ply
│   │   ...
└── realworld/
│   ├── fox/
│   │   ├── images
│   │   ├── images_generated
│   │   ├── point_cloud
│   │   ├── ...
│   │   └── moving_part_points.ply
│   │   ...
```

## 3. Running
```sh
# for synthetic dataset from PAC-NeRF
sh sim_synthetic.sh

# for real-world dataset from PhysDreamer
sh sim_physdreamer.sh

# for other real-world scenes
sh sim_realworld.sh
```


## 4. Custom Dataset
#### (1) Construct 3D Gaussian splats
To construct 3DGS, We recommend running [PGSR](https://github.com/zju3dv/PGSR) for multi-view images, [Grid4D](https://github.com/JiaweiXu8/Grid4D) for dynamic video, and [splatt3r](https://github.com/btsmart/splatt3r) for single image.

To isolate moving parts, we recommend [SAM2Point](https://github.com/ZiyuGuo99/SAM2Point) for automated segmentation or [supersplat](https://superspl.at/editor) for mannual segmentation.

#### (2) Generate video
Please refer to [video_generator.py](./video_generator.py) and modify the prompt text and input image according to your requirements.

#### (3) LLM-based material initialization
Please query LLMs (*e.g.*, GPT-4o) by providing the region of interest as visual input with the following query prompt:
> Q: What is this? What is the closest material type? Infer the density and corresponding material parameter values based on the material type.
>
> \* Note that the parameters used for different material types are as follows: $E, \nu$ for elasticity; $E, \nu, \tau_Y$ for plasticine and metal; $E, \nu, \eta$ for foam; $\theta_{fric}$ for sand; $\mu, \kappa$ for Newtonian fluid; $\mu, \kappa, \tau_Y, \eta$ for non-Newtonian fluid. The output should be in the following format: {"material_type": ... , "density": ...} .


For material types, we now support:
- ☑️ Elastic
- ☑️ Plasticine
- ☑️ Metal
- ☑️ Foam
- ☑️ Sand
- ☑️ Newtonian fluid
- ☑️ Non-Newtonian fluid

---
### Acknowledgements

This framework builds upon​​  [DreamPhysics](https://github.com/tyhuang0428/DreamPhysics) and [PhysGaussian](https://github.com/XPandora/PhysGaussian).​​ ​​For geometry acquisition​​, [PGSR](https://github.com/zju3dv/PGSR), [splatt3r](https://github.com/btsmart/splatt3r), and [Grid4D](https://github.com/JiaweiXu8/Grid4D) ​​served as fundamental geometric reconstruction components.​​ We sincerely appreciate the excellent works of these authors.

---

### Citation
If you find our work useful in your research, please cite:
```
@article{liu2025physflow,
  title={Unleashing the Potential of Multi-modal Foundation Models and Video Diffusion for 4D Dynamic Physical Scene Simulation},
  author={Liu, Zhuoman and Ye, Weicai and Luximon, Yan and Wan, Pengfei and Zhang, Di},
  journal={CVPR},
  year={2025}
}
```
