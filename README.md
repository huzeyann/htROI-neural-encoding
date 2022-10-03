# Upgrading Voxel-wise Encoding Model via Integrated Integration over Features and Brain Networks

Please see `notebooks/`. Full runtime steps: 

1. (Notebook000.ipynb) On whole-brain voxels. Launch multiple feature block concatenation baseline model with freeze backbone, record final score and use the 1/2 as backbone defrost milestone. 
1. (Notebook001.ipynb) On whole-brain voxels. Launch single layer-pooling feature block models (16 models for each backbone) with backbone defrost milestone score set according to step1.
1. (Notebook010.ipynb) Ensemble all backbone models from step2, also save their voxel embeddings.
1. (Notebook011.ipynb, Notebook012.ipynb) Run hierarchical clustering on voxel embeddings for each backbone model on their weighted and concatenated voxel embeddings. Save the resulting voxels clusters as htROI.
1. (Notebook100.ipynb) On ROI voxels, for each ROI. Launch multiple feature block concatenation baseline model with freeze backbone, record final score and use the 1/2 as backbone defrost milestone. 
1. (Notebook101.ipynb) On ROI voxels, for each ROI. Launch single layer-pooling feature block models (16 models for each ROI) with backbone defrost milestone score set according to step5.
1. (Notebook200.ipynb) Ensemble each ROI models from step6. Assemble ROIs from the same atlas to whole-brain. 
1. (Notebook200.ipynb) Do ROI intersection ensemble across atlas models.
2. (900~) subbmission to online evaluation on test set, and make figures

---

To setup the python env: 

```shell
conda create -n my_env python=3.8.8
conda activate my_env
#pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
pip install mmcv-full==1.4.6 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
pip install -r requirements.txt
# if tqdm_notebook is not displaying widgets
jupyter labextension install @jupyter-widgets/jupyterlab-manager

python setup.py install
```

<!-- ```shell
cp -r src/config/dataset/algonauts2021_roi_voxel_indexs /home/huze/Algonauts_2021_data/voxel_indexs
``` -->

Also update your paths in the `.env` file:
```
DATASET.ROOT_DIR=/home/huze/Algonauts_2021_data/
VOXEL_INDEX_DIR=/home/huze/Algonauts_2021_data/voxel_indexs/
MODEL.BACKBONE.PRETRAINED_WEIGHT_DIR=/home/huze/.cache/
TRAINER.CALLBACKS.CHECKPOINT.ROOT_DIR=/home/huze/.cache/checkpoints/
RESULTS_DIR=/data/huze/ray_results/algonauts2021/
```


If want to include optical flow model (I3d_Flow), optical flow need to be pre-computed:

> Note: ignore python env in `video_features`, use my python env 
```shell
git submodule update --init --recursive
cd video_features

find /home/huze/Algonauts_2021_data/raw/AlgonautsVideos268_All_30fpsmax/ -name '*.mp4' > path.txt
python main.py \
 --feature_type my \
 --file_with_video_paths path.txt \
 --device_ids 0 \
 --tmp_path /tmp \
 --on_extraction save_numpy \
 --output_path ./output \
 --extraction_fps 22 \
 --streams flow \
 --flow_type raft

mv ./output/my /home/huze/Algonauts_2021_data/precomputed_flow
cp models/i3d/checkpoints/i3d_flow.pt /home/huze/.cache/i3d_flow.pt

```