conda create -n mm_plan python=3.10 -y
conda activate mm_plan

cd verl
pip install --no-deps -e .

pip install -r requirements.txt
# flash attention (adjust the wheel URL for your CUDA/torch version)
wget https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.11/flash_attn-2.8.3+cu128torch2.8-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.8.3+cu128torch2.8-cp310-cp310-linux_x86_64.whl

pip install -r requirements_sglang.txt
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh

pip install pandas boto3 opencv-python qwen-vl-utils vllm matplotlib
