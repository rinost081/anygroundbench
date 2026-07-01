# Keep PyTorch aligned with the local CUDA 12.6 toolkit before
# building CUDA extensions such as flash-attn.
sudo apt-get update && sudo apt-get install -y ffmpeg

# clone the weight 
brew install git-xet
git xet install
git clone https://huggingface.co/bytedance-research/Vidi1.5-9B

# clone the vidi repository
git clone git@github.com:bytedance/vidi.git
cd vidi/Vidi1.5_9B/

# create a virtual environment (locked deps: pyproject.toml + uv.lock)
uv venv --python 3.10
source .venv/bin/activate
export MAX_JOBS=16
export NVCC_THREADS=4
uv sync

# python3 -u vidi/eval/inference.py --video-path [video path] --query [query] --model-path [model path]

# set the python path
export PYTHONPATH=/home/azureuser/data/vidi/Vidi1.5_9B:$PYTHONPATH

# run the inference
python3 -u vidi/eval/inference.py \
  --video-path /home/azureuser/data/bench/vidi_sample/demo_example_2.mp4 \
  --query "slicing onion" \
  --model-path /home/azureuser/data/Vidi1.5-9B