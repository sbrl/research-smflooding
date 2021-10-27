#!/usr/bin/env bash
#SBATCH -J TwImgCCT
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH -o %j.%N.%a.out
#SBATCH -e %j.%N.%a.err
#SBATCH -p gpu05,gpu
#SBATCH --time=1-04:00:00
#SBATCH --mem=25600


# 25600 = 25GiB memory required
module load utilities/multi
module load readline/7.0
module load gcc/10.2.0
module load cuda/11.5.0

module load python/anaconda/4.6/miniconda/3.7

CONFIG="${CONFIG:-configs/imageclassifier.toml}"
OUTPUT="${OUTPUT:-output/CHANGE_ME}";

MODE="FASHION_MNIST";

if [[ ! -r "${CONFIG}" ]]; then
	echo "Error: Config file at ${CONFIG} doesn't exist or we don't have permission to read it." >&2;
	exit 1;
fi

extra_flags="";
if [[ "${MODE}" == "FASHION_MNIST" ]]; then
	extra_flags="${extra_flags} --fashion-mnist";
fi


echo ">>> Installing requirements";
conda run -n py38 pip install -r requirements.txt;
echo ">>> Training model";
time conda run -n py38 src/image_classifier.py --only-gpu --config "${CONFIG}" --output "${OUTPUT}" ${extra_flags};
echo ">>> exited with code $?";
