# Time Lapse Mining


## Installation

We recommend using conda to install the required packages. The following commands will create a conda environment called `env_294` with all the required packages installed.

```
conda create -n env_294 python=3.8
conda activate env_294
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
conda install -c conda-forge imageio matplotlib pandas tqdm
pip install opencv-python
pip install flickrapi
```