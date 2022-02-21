python setup.py install
export CUDA_VISIBLE_DEVICES=0,1
python -m src.launch_grid \
 --exp-configs \
 src/config/experiments/algonauts2021_i3d_rgb.yml \
 src/config/experiments/algonauts2021_i3d_flow.yml \
 --schematics \
 multi_layer single_layer \
 --roi-config \
 src/config/dataset/algonauts2021_roi_defrost_score.json