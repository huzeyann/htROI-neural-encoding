python setup.py install
export CUDA_VISIBLE_DEVICES=0,1

python -m src.run_whole_brain -e src/config/experiments/algonauts2021/
