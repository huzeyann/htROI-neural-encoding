Laten ROI Voxel Encoding
```
src                              
├── config
│   ├── config.py                <- Default configs for the model 
│   └── experiments
│       ├── exp01_config.yaml    <- Configs for a specific experiment. Overwrites default 
│       └── exp02_config.yaml       configs
│       
├── data                  
│   ├── make_dataset.py          <- Script to generate data
│   ├── bengali_data.py          <- Custom Pytorch Dataset, DataLoader & Collator class
│   └── preprocessing.py         <- Custom data augmentation class
│
├── modeling                                  
│   ├── backbone                 <- Model backbone architecture
│   │   ├── se_resnext50.py
│   │   └── densenet121.py
│   │
│   ├── layers                   <- Custom layers
│   │   └── linear.py
│   │
│   ├── meta_arch                <- Scripts to combine backbone + head
│   │   ├── baseline.py
│   │   └── build.py
│   │
│   ├── head                     <- Build the head of the model
│   │   ├── build.py
│   │   └── simple_head.py
│   │
│   └── solver                   <- Scripts for building loss function, evaluation & optimizer
│       ├── loss
│       │   ├── build.py
│       │   ├── softmax_cross_entropy.py
│       │   └── label_smoothing_ce.py
│       ├── evaluation.py
│       └── optimizer.py 
│ 
├── tools                        <- Training loop and custom helper functions 
│   ├── train.py
│   └── registry.py 
│ 
└── visualization                <- Scripts for exploratory results & visualizations 
       └── visualize.py
```

```shell
rsync -avzP src/config/dataset/algonauts2021_roi_voxel_indexs
```

```shell
git submodule update --init --recursive
cd video_features

rsync -avzP models/i3d/checkpoints/i3d_flow.pt /home/huze/.cache/i3d_flow.pt

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
```

Note: ignore env config in `video_features`