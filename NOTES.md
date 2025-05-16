### Setting up conda environment - issues and solution

Running ‘conda env create -f environment.yaml’ gave the following terminal output:

C:\Users\cx24957\Documents\vfi\LDMVFI_new_env\LDMVFI>conda env create -f environment.yaml
Warning: you have pip-installed dependencies in your environment file, but you do not list pip itself as one of your conda dependencies.  Conda may not use the correct pip to install your packages, and they may end up in the wrong place.  Please add an explicit pip dependency.  I'm adding one for you, but still nagging you.
Channels:

- pytorch
- defaults
- conda-forge
Platform: win-64
Collecting package metadata (repodata.json): done
Solving environment: done

==> WARNING: A newer version of conda exists. <==
current version: 25.1.1
latest version: 25.3.0

Please update conda by running

```
$ conda update -n base -c defaults conda

```

Downloading and Extracting Packages:

Preparing transaction: done
Verifying transaction: done
Executing transaction: done
Installing pip dependencies: / Ran pip subprocess with arguments:
['C:\\Users\\cx24957\\AppData\\Local\\miniconda3\\envs\\ldmvfi_new_env\\python.exe', '-m', 'pip', 'install', '-U', '-r', 'C:\\Users\\cx24957\\Documents\\vfi\\LDMVFI_new_env\\LDMVFI\\condaenv.n_5653sx.re]
Pip subprocess output:
Obtaining taming-transformers from git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers (from -r C:\Users\cx24957\Documents\vfi\LDMVFI_new_env\LDMVFI\condaenv.n_5653sx.r)
Cloning https://github.com/CompVis/taming-transformers.git (to revision master) to c:\users\cx24957\documents\vfi\ldmvfi_new_env\ldmvfi\src\taming-transformers
Resolved https://github.com/CompVis/taming-transformers.git to commit 3ba01b241669f5ade541ce990f7650a3b8f65318
Preparing metadata ([setup.py](http://setup.py/)): started
Preparing metadata ([setup.py](http://setup.py/)): finished with status 'done'
Obtaining clip from git+https://github.com/openai/CLIP.git@main#egg=clip (from -r C:\Users\cx24957\Documents\vfi\LDMVFI_new_env\LDMVFI\condaenv.n_5653sx.requirements.txt (line 15))
Cloning https://github.com/openai/CLIP.git (to revision main) to c:\users\cx24957\documents\vfi\ldmvfi_new_env\ldmvfi\src\clip
Resolved https://github.com/openai/CLIP.git to commit dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1
Preparing metadata ([setup.py](http://setup.py/)): started
Preparing metadata ([setup.py](http://setup.py/)): finished with status 'done'
Obtaining file:///C:/Users/cx24957/Documents/vfi/LDMVFI_new_env/LDMVFI (from -r C:\Users\cx24957\Documents\vfi\LDMVFI_new_env\LDMVFI\condaenv.n_5653sx.requirements.txt (line 16))
Preparing metadata ([setup.py](http://setup.py/)): started
Preparing metadata ([setup.py](http://setup.py/)): finished with status 'done'
Collecting opencv-python==4.6.0.66 (from -r C:\Users\cx24957\Documents\vfi\LDMVFI_new_env\LDMVFI\condaenv.n_5653sx.requirements.txt (line 1))
Using cached opencv_python-4.6.0.66-cp36-abi3-win_amd64.whl.metadata (18 kB)
Collecting pudb==2022.1.3 (from -r C:\Users\cx24957\Documents\vfi\LDMVFI_new_env\LDMVFI\condaenv.n_5653sx.requirements.txt (line 2))
Using cached pudb-2022.1.3.tar.gz (220 kB)
Preparing metadata ([setup.py](http://setup.py/)): started
Preparing metadata ([setup.py](http://setup.py/)): finished with status 'done'
Collecting imageio==2.22.3 (from -r C:\Users\cx24957\Documents\vfi\LDMVFI_new_env\LDMVFI\condaenv.n_5653sx.requirements.txt (line 3))
Using cached imageio-2.22.3-py3-none-any.whl.metadata (5.0 kB)
Collecting imageio-ffmpeg==0.4.7 (from -r C:\Users\cx24957\Documents\vfi\LDMVFI_new_env\LDMVFI\condaenv.n_5653sx.requirements.txt (line 4))
Using cached imageio_ffmpeg-0.4.7-py3-none-win_amd64.whl.metadata (1.6 kB)
Collecting pytorch-lightning==1.7.7 (from -r C:\Users\cx24957\Documents\vfi\LDMVFI_new_env\LDMVFI\condaenv.n_5653sx.requirements.txt (line 5))
Using cached pytorch_lightning-1.7.7-py3-none-any.whl.metadata (27 kB)

Pip subprocess error:
Running command git clone --filter=blob:none --quiet https://github.com/CompVis/taming-transformers.git 'C:\Users\cx24957\Documents\vfi\LDMVFI_new_env\LDMVFI\src\taming-transformers'
Running command git clone --filter=blob:none --quiet https://github.com/openai/CLIP.git 'C:\Users\cx24957\Documents\vfi\LDMVFI_new_env\LDMVFI\src\clip'
WARNING: Ignoring version 1.7.7 of pytorch-lightning since it has invalid metadata:
Requested pytorch-lightning==1.7.7 from https://files.pythonhosted.org/packages/00/eb/3b2152f9c3a50d265f3e75529254228ace8a86e9a4397f3004f1e3be7825/pytorch_lightning-1.7.7-py3-none-any.whl (from -r C:\Uss    torch (>=1.9.*)
~~~~~~^
Please use pip<24.1 if you need to use this version.
ERROR: Could not find a version that satisfies the requirement pytorch-lightning==1.7.7 (from versions: 0.0.2, 0.2, 0.2.2, 0.2.3, 0.2.4, 0.2.4.1, 0.2.5, 0.2.5.1, 0.2.5.2, 0.2.6, 0.3, 0.3.1, 0.3.2, 0.3.3)ERROR: No matching distribution found for pytorch-lightning==1.7.7

failed
Running command git clone --filter=blob:none --quiet https://github.com/openai/CLIP.git 'C:\Users\cx24957\Documents\vfi\LDMVFI_new_env\LDMVFI\src\clip'
WARNING: Ignoring version 1.7.7 of pytorch-lightning since it has invalid metadata:
Requested pytorch-lightning==1.7.7 from https://files.pythonhosted.org/packages/00/eb/3b2152f9c3a50d265f3e75529254228ace8a86e9a4397f3004f1e3be7825/pytorch_lightning-1.7.7-py3-none-any.whl (from -r C:\Uss    torch (>=1.9.*)
~~~~~~^
Please use pip<24.1 if you need to use this version.
ERROR: Could not find a version that satisfies the requirement pytorch-lightning==1.7.7 (from versions: 0.0.2, 0.2, 0.2.2, 0.2.3, 0.2.4, 0.2.4.1, 0.2.5, 0.2.5.1, 0.2.5.2, 0.2.6, 0.3, 0.3.1, 0.3.2, 0.3.3)ERROR: No matching distribution found for pytorch-lightning==1.7.7

Running command git clone --filter=blob:none --quiet https://github.com/openai/CLIP.git 'C:\Users\cx24957\Documents\vfi\LDMVFI_new_env\LDMVFI\src\clip'
WARNING: Ignoring version 1.7.7 of pytorch-lightning since it has invalid metadata:
Requested pytorch-lightning==1.7.7 from https://files.pythonhosted.org/packages/00/eb/3b2152f9c3a50d265f3e75529254228ace8a86e9a4397f3004f1e3be7825/pytorch_lightning-1.7.7-py3-none-any.whl (from -r C:\Uss    torch (>=1.9.*)
~~~~~~^
Please use pip<24.1 if you need to use this version.
ERROR: Could not find a version that satisfies the requirement pytorch-lightning==1.7.7 (from versions: 0.0.2, 0.2, 0.2.2, 0.2.3, 0.2.4, 0.2.4.1, 0.2.5, 0.2.5.1, 0.2.5.2, 0.2.6, 0.3, 0.3.1, 0.3.2, 0.3.3)ERROR: No matching distribution found for pytorch-lightning==1.7.7
Running command git clone --filter=blob:none --quiet https://github.com/openai/CLIP.git 'C:\Users\cx24957\Documents\vfi\LDMVFI_new_env\LDMVFI\src\clip'
WARNING: Ignoring version 1.7.7 of pytorch-lightning since it has invalid metadata:
Requested pytorch-lightning==1.7.7 from https://files.pythonhosted.org/packages/00/eb/3b2152f9c3a50d265f3e75529254228ace8a86e9a4397f3004f1e3be7825/pytorch_lightning-1.7.7-py3-none-any.whl (from -r C:\Uss    torch (>=1.9.*)
~~~~~~^
Please use pip<24.1 if you need to use this version.
Running command git clone --filter=blob:none --quiet https://github.com/openai/CLIP.git 'C:\Users\cx24957\Documents\vfi\LDMVFI_new_env\LDMVFI\src\clip'
WARNING: Ignoring version 1.7.7 of pytorch-lightning since it has invalid metadata:
Requested pytorch-lightning==1.7.7 from https://files.pythonhosted.org/packages/00/eb/3b2152f9c3a50d265f3e75529254228ace8a86e9a4397f3004f1e3be7825/pytorch_lightning-1.7.7-py3-none-any.whl (from -r C:\Uss    torch (>=1.9.*)
~~~~~~^
Running command git clone --filter=blob:none --quiet https://github.com/openai/CLIP.git 'C:\Users\cx24957\Documents\vfi\LDMVFI_new_env\LDMVFI\src\clip'
WARNING: Ignoring version 1.7.7 of pytorch-lightning since it has invalid metadata:
Requested pytorch-lightning==1.7.7 from https://files.pythonhosted.org/packages/00/eb/3b2152f9c3a50d265f3e75529254228ace8a86e9a4397f3004f1e3be7825/pytorch_lightning-1.7.7-py3-none-any.whl (from -r C:\Uss    torch (>=1.9.*)
Running command git clone --filter=blob:none --quiet https://github.com/openai/CLIP.git 'C:\Users\cx24957\Documents\vfi\LDMVFI_new_env\LDMVFI\src\clip'
WARNING: Ignoring version 1.7.7 of pytorch-lightning since it has invalid metadata:
Requested pytorch-lightning==1.7.7 from https://files.pythonhosted.org/packages/00/eb/3b2152f9c3a50d265f3e75529254228ace8a86e9a4397f3004f1e3be7825/pytorch_lightning-1.7.7-py3-none-any.whl (from -r C:\Uss  Running command git clone --filter=blob:none --quiet https://github.com/openai/CLIP.git 'C:\Users\cx24957\Documents\vfi\LDMVFI_new_env\LDMVFI\src\clip'
WARNING: Ignoring version 1.7.7 of pytorch-lightning since it has invalid metadata:
Requested pytorch-lightning==1.7.7 from https://files.pythonhosted.org/packages/00/eb/3b2152f9c3a50d265f3e75529254228ace8a86e9a4397f3004f1e3be7825/pytorch_lightning-1.7.7-py3-none-any.whl (from -r C:\Uss  Running command git clone --filter=blob:none --quiet https://github.com/openai/CLIP.git 'C:\Users\cx24957\Documents\vfi\LDMVFI_new_env\LDMVFI\src\clip'
Running command git clone --filter=blob:none --quiet https://github.com/openai/CLIP.git 'C:\Users\cx24957\Documents\vfi\LDMVFI_new_env\LDMVFI\src\clip'
WARNING: Ignoring version 1.7.7 of pytorch-lightning since it has invalid metadata:
Requested pytorch-lightning==1.7.7 from https://files.pythonhosted.org/packages/00/eb/3b2152f9c3a50d265f3e75529254228ace8a86e9a4397f3004f1e3be7825/pytorch_lightning-1.7.7-py3-none-any.whl (from -r C:\Uss    torch (>=1.9.*)
Running command git clone --filter=blob:none --quiet https://github.com/openai/CLIP.git 'C:\Users\cx24957\Documents\vfi\LDMVFI_new_env\LDMVFI\src\clip'
WARNING: Ignoring version 1.7.7 of pytorch-lightning since it has invalid metadata:
Requested pytorch-lightning==1.7.7 from https://files.pythonhosted.org/packages/00/eb/3b2152f9c3a50d265f3e75529254228ace8a86e9a4397f3004f1e3be7825/pytorch_lightning-1.7.7-py3-none-any.whl (from -r C:\Uss  Running command git clone --filter=blob:none --quiet https://github.com/openai/CLIP.git 'C:\Users\cx24957\Documents\vfi\LDMVFI_new_env\LDMVFI\src\clip'
WARNING: Ignoring version 1.7.7 of pytorch-lightning since it has invalid metadata:
WARNING: Ignoring version 1.7.7 of pytorch-lightning since it has invalid metadata:
Requested pytorch-lightning==1.7.7 from https://files.pythonhosted.org/packages/00/eb/3b2152f9c3a50d265f3e75529254228ace8a86e9a4397f3004f1e3be7825/pytorch_lightning-1.7.7-py3-none-any.whl (from -r C:\Uss    torch (>=1.9.*)
~~~~~~^
Please use pip<24.1 if you need to use this version.
ERROR: Could not find a version that satisfies the requirement pytorch-lightning==1.7.7 (from versions: 0.0.2, 0.2, 0.2.2, 0.2.3, 0.2.4, 0.2.4.1, 0.2.5, 0.2.5.1, 0.2.5.2, 0.2.6, 0.3, 0.3.1, 0.3.2, 0.3.3)ERROR: No matching distribution found for pytorch-lightning==1.7.7

failed

CondaEnvException: Pip failed

Becuase of the error mmessage suggestion ‘Please use pip<24.1 if you need to use this version.’, I added pip version 24.0 to the dependencies in the environment.yaml. Fixed the original error message. 

C:\Users\cx24957\Documents\vfi\LDMVFI_new_env\LDMVFI>conda activate ldmvfi_new_env

(ldmvfi_new_env) C:\Users\cx24957\Documents\vfi\LDMVFI_new_env\LDMVFI>python [evaluate.py](http://evaluate.py/) --config configs/ldm/ldmvfi-vqflow-f32-c256-concat_max.yaml --ckpt ldm/models/checkpoints/ldmvfi-vqflow-f32-c256-concat_max.ckpt --dataset Ucf101_triplet --metrics PSNR SSIM --data_dir C:/Users/cx24957/Documents/vfi/datasets" --out_dir eval_results/ldmvfi-vqflow-f32-c256-concat_max --use_ddim
Traceback (most recent call last):
File "C:\Users\cx24957\Documents\vfi\LDMVFI_new_env\LDMVFI\[evaluate.py](http://evaluate.py/)", line 6, in <module>
(ldmvfi_new_env) C:\Users\cx24957\Documents\vfi\LDMVFI_new_env\LDMVFI>python [evaluate.py](http://evaluate.py/) --config configs/ldm/ldmvfi-vqflow-f32-c256-concat_max.yaml --ckpt ldm/models/checkpoints/ldmvfi-vqflow-f32-c256-concat_max.ckpt --dataset Ucf101_triplet --metrics PSNR SSIM --data_dir C:/Users/cx24957/Documents/vfi/datasets" --out_dir eval_results/ldmvfi-vqflow-f32-c256-concat_max --use_ddim
Traceback (most recent call last):
File "C:\Users\cx24957\Documents\vfi\LDMVFI_new_env\LDMVFI\[evaluate.py](http://evaluate.py/)", line 6, in <module>
56-concat_max --use_ddim
Traceback (most recent call last):
File "C:\Users\cx24957\Documents\vfi\LDMVFI_new_env\LDMVFI\[evaluate.py](http://evaluate.py/)", line 6, in <module>
Traceback (most recent call last):
File "C:\Users\cx24957\Documents\vfi\LDMVFI_new_env\LDMVFI\[evaluate.py](http://evaluate.py/)", line 6, in <module>
File "C:\Users\cx24957\Documents\vfi\LDMVFI_new_env\LDMVFI\[evaluate.py](http://evaluate.py/)", line 6, in <module>
from main import instantiate_from_config
File "C:\Users\cx24957\Documents\vfi\LDMVFI_new_env\LDMVFI\[main.py](http://main.py/)", line 6, in <module>
import pytorch_lightning as pl
File "C:\Users\cx24957\AppData\Local\miniconda3\envs\ldmvfi_new_env\lib\site-packages\pytorch_lightning\**init**.py", line 34, in <module>
from pytorch_lightning.callbacks import Callback  # noqa: E402
File "C:\Users\cx24957\AppData\Local\miniconda3\envs\ldmvfi_new_env\lib\site-packages\pytorch_lightning\callbacks\**init**.py", line 25, in <module>
from pytorch_lightning.callbacks.progress import ProgressBarBase, RichProgressBar, TQDMProgressBar
File "C:\Users\cx24957\AppData\Local\miniconda3\envs\ldmvfi_new_env\lib\site-packages\pytorch_lightning\callbacks\progress\**init**.py", line 22, in <module>
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar  # noqa: F401
File "C:\Users\cx24957\AppData\Local\miniconda3\envs\ldmvfi_new_env\lib\site-packages\pytorch_lightning\callbacks\progress\rich_progress.py", line 20, in <module>
from torchmetrics.utilities.imports import _compare_version
ImportError: cannot import name '_compare_version' from 'torchmetrics.utilities.imports' (C:\Users\cx24957\AppData\Local\miniconda3\envs\ldmvfi_new_env\lib\site-packages\torchmetrics\utilities\[imports.py](http://imports.py/))

Tried fixing this by running pip install torchmetrics==0.7.0, as suggested by a user on the issues page. 

Rerunning the code gives the error 

56-concat_max --use_ddim
RuntimeError: module compiled against ABI version 0x1000009 but this version of numpy is 0x2000000
Traceback (most recent call last):
File "C:\Users\cx24957\Documents\vfi\LDMVFI_new_env\LDMVFI\[evaluate.py](http://evaluate.py/)", line 8, in <module>
from ldm.data import testsets
File "C:\Users\cx24957\Documents\vfi\LDMVFI_new_env\LDMVFI\ldm\data\[testsets.py](http://testsets.py/)", line 10, in <module>
import utility
File "C:\Users\cx24957\Documents\vfi\LDMVFI_new_env\LDMVFI\[utility.py](http://utility.py/)", line 4, in <module>
import cv2
File "C:\Users\cx24957\AppData\Local\miniconda3\envs\ldmvfi_new_env\lib\site-packages\cv2\**init**.py", line 181, in <module>
bootstrap()
File "C:\Users\cx24957\AppData\Local\miniconda3\envs\ldmvfi_new_env\lib\site-packages\cv2\**init**.py", line 153, in bootstrap
native_module = importlib.import_module("cv2")
File "C:\Users\cx24957\AppData\Local\miniconda3\envs\ldmvfi_new_env\lib\importlib\**init**.py", line 127, in import_module
return _bootstrap._gcd_import(name[level:], package, level)
ImportError: numpy.core.multiarray failed to import

I think this is related to numpy being version 2.0.1, but opencv code being compiled with numpy version 1.x.x. I tried downgrading numpy to version 1.26.4.

Next got this error:

Traceback (most recent call last):
File "C:\Users\cx24957\Documents\vfi\LDMVFI_new_env\LDMVFI\[evaluate.py](http://evaluate.py/)", line 8, in <module>
from ldm.data import testsets
File "C:\Users\cx24957\Documents\vfi\LDMVFI_new_env\LDMVFI\ldm\data\[testsets.py](http://testsets.py/)", line 14, in <module>
from ldm.models.autoencoder import *
File "C:\Users\cx24957\Documents\vfi\LDMVFI_new_env\LDMVFI\ldm\models\[autoencoder.py](http://autoencoder.py/)", line 12, in <module>
from ldm.modules.diffusionmodules.model import *
File "C:\Users\cx24957\Documents\vfi\LDMVFI_new_env\LDMVFI\ldm\modules\diffusionmodules\[model.py](http://model.py/)", line 10, in <module>
from cupy_module import dsepconv
File "C:\Users\cx24957\Documents\vfi\LDMVFI_new_env\LDMVFI\cupy_module\[dsepconv.py](http://dsepconv.py/)", line 7, in <module>
class Stream:
File "C:\Users\cx24957\Documents\vfi\LDMVFI_new_env\LDMVFI\cupy_module\[dsepconv.py](http://dsepconv.py/)", line 8, in Stream
ptr = torch.cuda.current_stream().cuda_stream
File "C:\Users\cx24957\AppData\Local\miniconda3\envs\ldmvfi_new_env\lib\site-packages\torch\cuda\**init**.py", line 1010, in current_stream
_lazy_init()
File "C:\Users\cx24957\AppData\Local\miniconda3\envs\ldmvfi_new_env\lib\site-packages\torch\cuda\**init**.py", line 310, in _lazy_init
raise AssertionError("Torch not compiled with CUDA enabled")
AssertionError: Torch not compiled with CUDA enabled

i tried following the instructions for installing pytorch with cuda enabled from the website, ran ‘pip3 install torch==1.11.0 torchvision=0.12.0 --index-url https://download.pytorch.org/whl/cu118 ’. Output was:

ERROR: Could not find a version that satisfies the requirement torch==1.11.0 (from versions: 2.0.0+cu118, 2.0.1+cu118, 2.1.0+cu118, 2.1.1+cu118, 2.1.2+cu118, 2.2.0+cu118, 2.2.1+cu118, 2.2.2+cu118, 2.3.0+cu118, 2.3.1+cu118, 2.4.0+cu118, 2.4.1+cu118, 2.5.0+cu118, 2.5.1+cu118, 2.6.0+cu118)
ERROR: No matching distribution found for torch==1.11.0

Tried installing without version specifications. This installed, but still gave the ‘ Torch not compiled with CUDA enabled’ error.  Next attempt was to create the conda environment, but with the config.yaml file oncluding the changes made (pip dependency, numpy version and torchmetrics version). All seems to be running 

### getting weights for lpips model

In the class constructor for LPIPS, code tried to load a state_dict from a location that wasn’t there. I downloaded the weights from the repo for the lpips paper and put the in weights/v0.1 like the code was expecting.

### Also needed to install taming transformers directly using pip after the conda env was created successfully. 

### needed to change cupy dependency to cupy-cuda113 to work for wsl