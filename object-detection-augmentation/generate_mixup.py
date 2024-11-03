import os
from random import sample


import numpy as np
from PIL import Image, ImageDraw

from utils.random_data import get_random_data, get_random_data_with_MixUp,get_random_clean_data
from utils.utils import convert_annotation, get_classes

#-----------------------------------------------------------------------------------#
#   Origin_VOCdevkit_path   原始数据集所在的路径
#   Out_VOCdevkit_path      输出数据集所在的路径
#-----------------------------------------------------------------------------------#
Origin_VOCdevkit_path   = "D:/xray/object-detection-augmentation/VOCdevkit_Origin"
Out_VOCdevkit_path      = "D:/xray/object-detection-augmentation/VOCdevkit"
#-----------------------------------------------------------------------------------#
#   Out_Num                 利用mixup生成多少组图片
#   input_shape             生成的图片大小
#-----------------------------------------------------------------------------------#
Out_Num                 = 600
input_shape             = [600, 600]

#-----------------------------------------------------------------------------------#
#   下面定义了xml里面的组成模块，无需改动。
#-----------------------------------------------------------------------------------#
headstr = """\
<annotation>
    <folder>VOC</folder>
    <filename>%s</filename>
    <source>
        <database>My Database</database>
        <annotation>COCO</annotation>
        <image>flickr</image>
        <flickrid>NULL</flickrid>
    </source>
    <owner>
        <flickrid>NULL</flickrid>
        <name>company</name>
    </owner>
    <size>
        <width>%d</width>
        <height>%d</height>
        <depth>%d</depth>
    </size>
    <segmented>0</segmented>
"""

objstr = """\
    <object>
        <name>%s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>%d</xmin>
            <ymin>%d</ymin>
            <xmax>%d</xmax>
            <ymax>%d</ymax>
        </bndbox>
    </object>
"""
    
tailstr = '''\
</annotation>
'''
if __name__ == "__main__":
    Origin_JPEGImages_path  = os.path.join(Origin_VOCdevkit_path, "VOC2007/JPEGImages")
    Origin_Annotations_path = os.path.join(Origin_VOCdevkit_path, "VOC2007/Annotations")

    Origin_JPEGImages_path_clean = os.path.join(Origin_VOCdevkit_path, "VOC2007/JPEGImages_clean")
    Origin_Annotations_path_clean = os.path.join(Origin_VOCdevkit_path, "VOC2007/Annotations_clean")
    
    Out_JPEGImages_path  = os.path.join(Out_VOCdevkit_path, "VOC2007/JPEGImages")
    Out_Annotations_path = os.path.join(Out_VOCdevkit_path, "VOC2007/Annotations")
    
    if not os.path.exists(Out_JPEGImages_path):
        os.makedirs(Out_JPEGImages_path)
    if not os.path.exists(Out_Annotations_path):
        os.makedirs(Out_Annotations_path)
    #---------------------------#
    #   遍历标签并赋值
    #---------------------------#
    xml_names = os.listdir(Origin_Annotations_path)
    xml_name_clean = os.listdir(Origin_Annotations_path_clean)

    def write_xml(anno_path, jpg_pth, head, input_shape, boxes, unique_labels, tail):
        f = open(anno_path, "w")
        f.write(head%(jpg_pth, input_shape[0], input_shape[1], 3))
        for i, box in enumerate(boxes):
            f.write(objstr%(str(unique_labels[int(box[4])]), box[0], box[1], box[2], box[3]))
        f.write(tail)
    
    #------------------------------#
    #   循环生成xml和jpg
    #------------------------------#
    from tqdm import tqdm
    for index in tqdm(range(Out_Num),desc='Processing'):
        #------------------------------#
        #   获取两个图像与标签
        #------------------------------#
        sample_xmls = []
        sample_xmls_back = sample(xml_names, 1)
        sample_xmls_clean = sample(xml_name_clean, 1)
        sample_xmls = [sample_xmls_back[0], sample_xmls_clean[0]]

        unique_labels = get_classes(sample_xmls, Origin_Annotations_path,Origin_Annotations_path_clean)

        jpg_name_1  = os.path.join(Origin_JPEGImages_path, os.path.splitext(sample_xmls_back[0])[0] + '.jpg')
        jpg_name_2  = os.path.join(Origin_JPEGImages_path_clean, os.path.splitext(sample_xmls_clean[0])[0] + '.jpg')
        
        xml_name_1  = os.path.join(Origin_Annotations_path, sample_xmls_back[0])
        xml_name_2  = os.path.join(Origin_Annotations_path_clean, sample_xmls_clean[0])

        width,height = Image.open(jpg_name_1).size
        back_size = [height,width]
        

        
            
        line_1 = convert_annotation(jpg_name_1, xml_name_1, unique_labels)
        line_2 = convert_annotation(jpg_name_2, xml_name_2, unique_labels)
        
        #------------------------------#
        #   各自数据增强
        #------------------------------#
        # image_1, box_1  = get_random_data(line_1, back_size) 
        image_1, box_1  = get_random_data(line_1, back_size) 
        image_2, box_2  = get_random_clean_data(line_2, back_size) 
        
        #------------------------------#
        #   合并mixup
        #------------------------------#
        image_data, box_data = get_random_data_with_MixUp(image_1, box_1, image_2, box_2)
        
        img = Image.fromarray(image_data.astype(np.uint8))
        img.save(os.path.join(Out_JPEGImages_path, 'mix20240915_'+ str(index) + '.jpg'))
        write_xml(os.path.join(Out_Annotations_path, 'mix20240915_'+ str(index) + '.xml'), os.path.join(Out_JPEGImages_path, 'mix20240826_'+str(index) + '.jpg'), \
                    headstr, input_shape, box_data, unique_labels, tailstr)