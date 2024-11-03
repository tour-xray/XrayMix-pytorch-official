import copy
import os
import xml.etree.ElementTree as ET

# def get_classes(sample_xmls, Origin_Annotations_path):
#     unique_labels  = []
#     for xml in sample_xmls:
#         in_file = open(os.path.join(Origin_Annotations_path, xml), encoding='utf-8')
#         tree    = ET.parse(in_file)
#         root    = tree.getroot()
        
#         for obj in root.iter('object'):
#             cls     = obj.find('name').text
#             if cls not in unique_labels:
#                 unique_labels.append(cls)
#     return unique_labels

def get_classes(sample_xmls, path1 , path2):
    unique_labels  = []
    in_file1 = open(os.path.join(path1, sample_xmls[0]), encoding='utf-8')
    tree1    = ET.parse(in_file1)
    root1    = tree1.getroot()
    for obj in root1.iter('object'):
        cls     = obj.find('name').text
        if cls not in unique_labels:
            unique_labels.append(cls)

    in_file2 = open(os.path.join(path2, sample_xmls[1]), encoding='utf-8')
    tree2    = ET.parse(in_file2)
    root2    = tree2.getroot()
    for obj in root2.iter('object'):
        cls     = obj.find('name').text
        if cls not in unique_labels:
            unique_labels.append(cls)

    return unique_labels

def get_classes_backnolabel(sample_xmls, path):
    unique_labels  = []
    # in_file1 = open(os.path.join(path1, sample_xmls[0]), encoding='utf-8')
    # tree1    = ET.parse(in_file1)
    # root1    = tree1.getroot()
    # for obj in root1.iter('object'):
    #     cls     = obj.find('name').text
    #     if cls not in unique_labels:
    #         unique_labels.append(cls)
    in_file = open(os.path.join(path, sample_xmls[0]), encoding='utf-8')
    tree2    = ET.parse(in_file)
    root2    = tree2.getroot()
    for obj in root2.iter('object'):
        cls     = obj.find('name').text
        if cls not in unique_labels:
            unique_labels.append(cls)

    return unique_labels

def convert_annotation(jpg_path, xml_path, classes):
    line = copy.deepcopy(jpg_path)
    if xml_path == None:
        line += " "
        return line
    else:
        in_file = open(xml_path, encoding='utf-8')
        tree    = ET.parse(in_file)
        root    = tree.getroot()
    
    
        for obj in root.iter('object'):
            difficult = 0 
            if obj.find('difficult')!=None and hasattr(obj, "text"):
                difficult = obj.find('difficult').text
            if int(difficult)==1:
                continue
        
            cls     = obj.find('name').text
            cls_id = classes.index(cls)
        
            xmlbox  = obj.find('bndbox')
            b       = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        
            line += " " + ",".join([str(a) for a in b]) + ',' + str(cls_id)
        return line