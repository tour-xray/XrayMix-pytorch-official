# 🤯XrayMix-Pytorch-OFFICIAL
👏 XrayMix implementation and SANDet source code<br>
**[2024/11/03]** Updated Readme.md, and released the source code of XrayMix. Please note that this version of the source code uses pytorch refactoring and is not directly trainable for target detection tasks. The source code for XrayMix, which is integrated with the mmdetection framework, will be released soon.
 
## 🤯Catalogs

- [Start](#Start)
- [Document](#Document)
- [Frameworks](#Frameworks)
- [Contributor](#Contributor)
- [Acknowledge](#Acknowledge)

### 🤯Start

Required environment
```sh
pytorch >= 1.12.1
mmcv == 2.1.0
mmengine == 0.10.1
```


```sh
git clone https://github.com/tour-xray/XrayMix-pytorch-official.git
```

XrayMix processes two random labeled images:
```sh
cd object-detection-augmentation
python generate_mixup.py
```

You need to put the two images into the VOC folder in batches, the xml files need to correspond and set the output image size by yourself. We have integrated XrayMix into the mmdet framework (Source code will be avaliable soon).

### 🤠Document

```
To be updated
```

### 🤠Frameworks

- [mmdetection](https://github.com/open-mmlab/mmdetection)

### 🤠DataSets
[PIDray](https://github.com/bywang2018/security-dataset)<br>
The PIDray we used is the version published in ICCV 2021.<br>
<br>
[114Xray](https://github.com/ming076/114Xray.)<br>
The 114Xray we used is the version published in PRCV 2024.<br>


### 🤠Contributor

Anonymous.


### 🤠Author
Anonymous.


### 🤠Acknowledge


- [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
- [bubbliiiing](https://github.com/bubbliiiing/object-detection-augmentation)
- [Openmmlab](https://github.com/open-mmlab/mmdetection)

