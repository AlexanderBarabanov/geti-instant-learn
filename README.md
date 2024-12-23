# Visual Prompting Exploration

## Data

Download the PerSeg dataset 
Please download our constructed dataset PerSeg for personalized segmentation from [Google Drive](https://drive.google.com/file/d/18TbrwhZtAPY5dlaoEqkPa5h08G9Rjcio/view?usp=sharing) and the pre-trained weights of SAM from [Meta Research](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth). 

Unzip the dataset and organize the data as:
```
data/
├── PerSeg/
│   ├── Images/
│   ├── Annotations/
├── sam_vit_h_4b8939.pth
```

Download [480p TrainVal split of DAVIS 2017](). Then decompress the file and place the directory in data like this:
```
data/
├── DAVIS/
│   ├── 2017/
│   │   ├── JPEGImages/
│   │   ├── Annotations/
│   │   ├── ImageSets/
│   │   │   ├── 2016/
│   │   │   ├── 2017/
```

Copy the pre-trained weights of mobileSAM from PersonalizeSAM/weights/mobile_sam.pt to data/mobile_sam.pt


## Installation

Create a new conda/mamba/venv environment and install the dependencies.

```
conda create -n visualprompting python=3.11
conda activate visualprompting
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

pip install -r requirements.txt
```


## Run

Either use the main.py to run the code or use the gradio interface. 

```
python main.py
# or
python ui.py
```

When using the gradio interface make sure to tunnel the port to your local machine when your are running the code on a remote server.

```
ssh -L 8888:localhost:8888 username@server_ip
```

