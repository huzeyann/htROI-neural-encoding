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