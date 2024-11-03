# 🤯XrayMix-Offical
👏 XrayMix implementation and SANDet source code<br>
**[2024/11/03]** Updated Readme.md, and released the source code of XrayMix.
 
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
python generate_mixup.py
```

You need to put the two images into the VOC folder in batches, the xml files need to correspond and set the output image size by yourself.
We have integrated XrayMix into the mmdet framework, and using XrayMix for training is simply a matter of using the Mixup data augmentation, as described in our published config file and in **mmdetection/mmdet/datasets/transforms/transforms.py** and in the **mmdetection/mmdet/datasets/transforms/XrayMix.py**.


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
[114Xray]()<br>
The 114Xray we used is the version published in PRCV 2024.<br>


### 🤠Contributor

Li Litao(SCUT)


### 🤠Author

mail: 202320116452@mail.scut.edu.cn

 *If you have questions that are not answered in a timely manner, you can contact us by email at 202320116452@mail.scut.edu.cn*

### 🤠License

The project is licensed under the MIT License, for more information see [LICENSE.txt]()

### 🤠Acknowledge


- [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
- [bubbliiiing](https://github.com/bubbliiiing/object-detection-augmentation)
- [Openmmlab](https://github.com/open-mmlab/mmdetection)

