python setup.py install
export CUDA_VISIBLE_DEVICES=0,1

python -m src.launch_grid \
 --exp-config \
 src/config/experiments/algonauts2021_i3d_rgb.yml \
 --schematic \
 single_layer \
 --roi-config \
 src/config/dataset/algonauts2021_i3d_rgb_defrost_score.json
python -m src.launch_grid \
 --exp-config \
 src/config/experiments/algonauts2021_i3d_flow.yml \
 --schematic \
 single_layer \
 --roi-config \
 src/config/dataset/algonauts2021_i3d_flow_defrost_score.json
python -m src.launch_grid \
 --exp-config \
 src/config/experiments/algonauts2021_i3d_rgb.yml \
 --schematic \
 multi_layer \
 --roi-config \
 src/config/dataset/algonauts2021_i3d_rgb_defrost_score.json
python -m src.launch_grid \
 --exp-config \
 src/config/experiments/algonauts2021_i3d_flow.yml \
 --schematic \
 multi_layer \
 --roi-config \
 src/config/dataset/algonauts2021_i3d_flow_defrost_score.json