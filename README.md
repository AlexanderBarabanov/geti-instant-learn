# Visual Prompting Exploration

This project explores new algorithms and backbones for the Visual Prompting service currently used in Intel Geti. It provides a framework to evaluate different visual prompting approaches, including:

- Various algorithms (SAM, P2SAM, ModelAPI implementation)
- Different backbone architectures (MobileSAM, EfficientViT-SAM, OpenVino optimized MobileSAM, etc.)
- Multiple datasets for comprehensive evaluation

Key features:
- Unified evaluation pipeline to benchmark algorithms and backbones
- Interactive Gradio UI for visual inspection and algorithm comparison
- Support for the existing ModelAPI implementation
- Automated metrics calculation (mIoU, mAcc) across datasets
- Easy integration of new algorithms and backbones

The project aims to identify improvements and alternatives to the current visual prompting approach used in production.

Note that this project is work in progress and is not meant to be used in production.


## Data

Download all data from the [sharepoint link](https://intel.sharepoint.com/sites/WorkflowPlatform/Shared%20Documents/Forms/AllItems.aspx?ct=1736256097696&or=Teams%2DHL&ga=1&id=%2Fsites%2FWorkflowPlatform%2FShared%20Documents%2FVisualPrompting%2FVPS%20data%2Ezip&parent=%2Fsites%2FWorkflowPlatform%2FShared%20Documents%2FVisualPrompting)

Unzip the file, and put the "data" directory in your user home folder:
```
~/data/
├── PerSeg/
│   ├── Images/
│   ├── Annotations/
├── sam_vit_h_4b8939.pth
...
```

Download the Efficient-ViT-SAM model checkpoint from this [sharepoint link](https://intel.sharepoint.com/:u:/r/sites/WorkflowPlatform/Shared%20Documents/VisualPrompting/data/efficientvit_sam_l0.pt?csf=1&web=1&e=WzoHyN) and place the file in `/efficientvit/assets/checkpoints/efficientvit_sam_l0.pt`. 


## Installation

Create a new conda/mamba/venv environment and install the dependencies.

```
conda create -n visualprompting python=3.11
conda activate visualprompting
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

pip install -r requirements.txt
```

Install custom Model API package:
```
pip install -e model_api/model_api/python
```

Install custom backbones:
```
pip install -e efficientvit
```


## Run

Either use main.py to run the evaluation scripts or use the gradio interface for interactive testing. 

```
python main.py
# or
python ui.py
```

When using the gradio interface make sure to tunnel the port to your local machine when your are running the code on a remote server.

```
ssh -L 8888:localhost:8888 username@server_ip
```

