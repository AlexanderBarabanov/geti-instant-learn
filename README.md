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

