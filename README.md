# Modeling Indirect Illumination for Inverse Rendering

### [Project Page](https://zju3dv.github.io/invrender) | [Paper](https://arxiv.org/pdf/2204.06837.pdf) | [Data](https://drive.google.com/file/d/1wWWu7EaOxtVq8QNalgs6kDqsiAm7xsRh/view?usp=sharing)



## Preparation
- Set up the python environment

```sh
docker build \
   --build-arg http_proxy=${http_proxy} \
   -t inv_render \
   -f docker/Dockerfile .                                      
```


```sh
conda create -n invrender python=3.7
conda activate invrender

pip install -r requirement.txt
```

- Dowload [DTUMVS dataset](https://www.dropbox.com/sh/5tam07ai8ch90pf/AADniBT3dmAexvm_J1oL__uoa) used in [IDR work](https://github.com/lioryariv/idr)


## Run the code

#### Training

Taking the scene `scan69` as an example, the training process is as follows.

1. Optimize geometry and outgoing radiance field from multi-view images. (Same as [IDR](https://github.com/lioryariv/idr))

   ```sh
   cd code
   python3 training/exp_runner.py --conf confs_sg/dtumvs.conf \
                                  --data_dir ../DTU \
                                  --scan_id 69 \
                                  --trainstage IDR \
                                  --gpu 1
   ```

2. Draw sample rays above surface points to train the indirect illumination and visibility MLP.

   ```sh
   python3 training/exp_runner.py --conf confs_sg/dtumvs.conf \
                                  --data_dir ../DTU \
                                  --scan_id 69 \
                                  --trainstage Illum \
                                  --gpu 1
   ```
   
3. Jointly optimize diffuse albedo, roughness and direct illumination.

   ```sh
   python3 training/exp_runner.py --conf confs_sg/dtumvs.conf \
                                  --data_dir ../DTU \
                                  --scan_id 69 \
                                  --trainstage Material \
                                  --gpu 1
   ```

#### Relighting

- Generate videos under novel illumination.

  ```sh
  python3 scripts/relight.py --conf confs_sg/dtumvs.conf \
                             --data_dir ../DTU \
                             --scan_id 69 \
                             --timestamp latest \
                             --gpu 1
  ```

#### Extracting

- Extract mesh with material

  ```sh
  python3 scripts/extract.py --conf confs_sg/dtumvs.conf \
                             --scan_id 69 \
                             --timestamp latest \
                             --gpu 1
  ```  

## Citation

```
@inproceedings{zhang2022invrender,
  title={Modeling Indirect Illumination for Inverse Rendering},
  author={Zhang, Yuanqing and Sun, Jiaming and He, Xingyi and Fu, Huan and Jia, Rongfei and Zhou, Xiaowei},
  booktitle={CVPR},
  year={2022}
}
```

Acknowledgements: part of our code is inherited from  [IDR](https://github.com/lioryariv/idr) and [PhySG](https://github.com/Kai-46/PhySG). We are grateful to the authors for releasing their code.

