
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
from prepare import Prep
from preprocess_utils import HUnorm, resample
from medicalseg.utils import wrapped_partial

class Prep_CT_spine(Prep):
    def __init__(self):
        super().__init__(
            dataset_root="/home/aistudio/work/", #用来存放转换后的npy数据
            raw_dataset_dir="/home/aistudio/work/", #假如用绝对路径就无视dataset_root
            images_dir="SpleenAndLiver/train/data",#训练集文件路径
            labels_dir="SpleenAndLiver/train/mask",#训练集mask文件路径
            phase_dir="SpleenAndLiver_Np/",#存放转换后的numpy文件
            images_dir_test="SpleenAndLiver/test",#测试集文件路径
            valid_suffix=("nii.gz", "nii.gz"))

        self.preprocess = {
            "images": [
                wrapped_partial(
                    HUnorm, HU_min=-100, HU_max=300),#设置窗宽窗位的裁剪范围
                wrapped_partial(
                    resample, new_shape=[128,128,128], order=1)#设置输入网络的数据形状，顺序是[z,y,x]
            ],
            "labels": [
                wrapped_partial(
                    resample, new_shape=[128,128,128], order=0),
            ],
            "images_test":[
                wrapped_partial(
                    HUnorm, HU_min=-100, HU_max=300),
                wrapped_partial(
                    resample, new_shape=[128, 128, 128], order=1)
            ]
        }
    def generate_txt(self, train_split=0.8):
        """generate the train_list.txt and val_list.txt
           0.8是训练集和测试集的分割比例，训练集占9份
        """
        txtname = [
            os.path.join(self.phase_path, 'train_list.txt'),
            os.path.join(self.phase_path, 'val_list.txt')
        ]
        image_files_npy = os.listdir(self.image_path)
        label_files_npy = [
            #原始文件和mask文件的名字差别seg字符。因此可以根据原文件的名字修改得到mask文件的名字。
            name.replace("_0000.npy", ".npy") for name in image_files_npy
        ]
        self.split_files_txt(txtname[0], image_files_npy, label_files_npy,
                             train_split)
        self.split_files_txt(txtname[1], image_files_npy, label_files_npy,
                             train_split)

if __name__ == "__main__":
    prep = Prep_CT_spine()
    prep.generate_dataset_json(
            modalities=('CT', ),
            #设置标签名称
            labels={
                0: "Background",
                1: "liver",
                2: "spleen",
            },
            dataset_name="spleenAndLiver Seg",#设置数据相关的描述
            dataset_description="###",
            license_desc="###",
            dataset_reference="###", )
    prep.load_save()
    prep.generate_txt()
