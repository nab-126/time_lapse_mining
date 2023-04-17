# Time Lapse Mining

Please see this [link](https://inst.eecs.berkeley.edu/~cs194-26/fa22/upload/files/projFinalProposed/cs194-26-aeg/Nabeel_Hingun_FinalProposedProject/Time-lapse%20Mining%20from%20Internet%20Photos%201931f579ceea4e3dba8734cb201e356f.html) for the project results.

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

For SfM reconstruction, you will also need to install COLMAP. Follow the instructions here:
https://colmap.github.io/install.html

