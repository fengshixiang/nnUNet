import numpy as np
from collections import OrderedDict

from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import nnUNet_raw_data
import SimpleITK as sitk
import shutil
import os


def copy_Multiorgan_segmentation_and_convert_labels(in_file, out_file):
    # use this for segmentation only!!!
    # nnUNet wants the labels to be continuous. 
    # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14
    # -> we make that into   0, 1, 3, 4, 5, 6, 7, 11, 14
    # -> then make that into 0, 1, 2, 3, 4, 5, 6, 7, 8

    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)

    uniques = np.unique(img_npy)
    for u in uniques:
        if u not in list(range(15)):
            raise RuntimeError('unexpected label')

    seg_new = np.zeros_like(img_npy)
    seg_new[img_npy == 1] = 1
    seg_new[img_npy == 3] = 2
    seg_new[img_npy == 4] = 3
    seg_new[img_npy == 5] = 4
    seg_new[img_npy == 6] = 5
    seg_new[img_npy == 7] = 6
    seg_new[img_npy == 11] = 7
    seg_new[img_npy == 14] = 8
    img_corr = sitk.GetImageFromArray(seg_new)
    img_corr.CopyInformation(img)
    sitk.WriteImage(img_corr, out_file)


def copy_Multiorgan_segmentation_and_convert_labels_inverse(in_file, out_file):
    # use this for segmentation only!!!
    # nnUNet wants the labels to be continuous. 
    # 0, 1, 2, 3, 4, 5, 6, 7, 8
    # 0, 1, 3, 4, 5, 6, 7, 11, 14

    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)

    uniques = np.unique(img_npy)
    for u in uniques:
        if u not in list(range(15)):
            raise RuntimeError('unexpected label')

    seg_new = np.zeros_like(img_npy)
    seg_new[img_npy == 1] = 1
    seg_new[img_npy == 2] = 3
    seg_new[img_npy == 3] = 4
    seg_new[img_npy == 4] = 5
    seg_new[img_npy == 5] = 6
    seg_new[img_npy == 6] = 7
    seg_new[img_npy == 7] = 11
    seg_new[img_npy == 8] = 14
    img_corr = sitk.GetImageFromArray(seg_new)
    img_corr.CopyInformation(img)
    sitk.WriteImage(img_corr, out_file)


def main():
    task_name = "Task101_MultiOrgan"
    downloaded_data_dir = "/DATA2/data/yuhangzhou"

    target_base = join(nnUNet_raw_data, task_name)
    target_imagesTr = join(target_base, "imagesTr")
    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTr = join(target_base, "labelsTr")

    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTr)

    index = 1

    # tcia data
    tcia_img_dir = join(downloaded_data_dir, 'Pancreas-CT/Pancreas_nii')
    tcia_label_dir = join(downloaded_data_dir, 'label_tcia_multiorgan')

    patient_names = []
    tcia_patient_list = os.listdir(tcia_label_dir)
    tcia_patient_list.sort()
    for p in tcia_patient_list:
        seg = join(tcia_label_dir, p)
        pat_index = p.split('.')[0][-4:]
        ct = join(tcia_img_dir, 'Pancreas_{}.nii.gz'.format(pat_index))
        patient_name = 'multiorgan_{:0>3d}'.format(index)

        if int(pat_index) != 25:
            assert all([
                isfile(seg),
                isfile(ct),
            ]), "%s" % pat_index
            patient_names.append(patient_name)
            print(pat_index, index)
            shutil.copy(ct, join(target_imagesTr, patient_name + "_0000.nii.gz"))
            # shutil.copy(seg, join(target_labelsTr, patient_name + ".nii.gz"))
            copy_Multiorgan_segmentation_and_convert_labels(seg, join(target_labelsTr,  patient_name + ".nii.gz"))
        index += 1

    # btcv data
    btcv_img_dir = join(downloaded_data_dir, 'Abdomen/RawData')
    btcv_label_dir = join(downloaded_data_dir, 'label_btcv_multiorgan')
    btcv_patient_list = os.listdir(btcv_label_dir)
    btcv_patient_list.sort()
    for p in btcv_patient_list:
        seg = join(btcv_label_dir, p)
        pat_index = p.split('.')[0][-4:]
        if int(pat_index) <= 40: # traing
            ct = join(btcv_img_dir, "Training/img", 'img{}.nii.gz'.format(pat_index))
        else:
            ct = join(btcv_img_dir, "Testing/img", 'img{}.nii.gz'.format(pat_index))
        patient_name = 'multiorgan_{:0>3d}'.format(index)
        patient_names.append(patient_name)

        assert all([
            isfile(seg),
            isfile(ct),
        ]), "%s" % pat_index
        print(pat_index, index)
        shutil.copy(ct, join(target_imagesTr, patient_name + "_0000.nii.gz"))
        # shutil.copy(seg, join(target_labelsTr, patient_name + ".nii.gz"))
        copy_Multiorgan_segmentation_and_convert_labels(seg, join(target_labelsTr,  patient_name + ".nii.gz"))
        index += 1

    json_dict = OrderedDict()
    json_dict['name'] = "MultiOrgan"
    json_dict['description'] = "nothing"
    json_dict['tensorImageSize'] = "3D"
    json_dict['reference'] = "see ***"
    json_dict['licence'] = "see ***"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT"
    }
    # json_dict['labels'] = {
    #     "0": "background",
    #     "1": "spleen",
    #     "2": "right kidney",
    #     "3": "left kidney",
    #     "4": "gallbladder",
    #     "5": "esophagus",
    #     "6": "liver",
    #     "7": "stomach",
    #     "8": "aorta",
    #     "9": "inferior vena cava",
    #     "10": "portal vein and splenic vein",
    #     "11": "pancreas",
    #     "12": "right adrenal gland",
    #     "13": "left adrenal gland",
    #     "14": "duodenum",
    # }
    json_dict['labels'] = {
        "0": "background",
        "1": "spleen",
        "2": "left kidney",
        "3": "gallbladder",
        "4": "esophagus",
        "5": "liver",
        "6": "stomach",
        "7": "pancreas",
        "8": "duodenum",
    }
    json_dict['numTraining'] = len(patient_names)
    json_dict['numTest'] = 0
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                             patient_names]
    json_dict['test'] = []

    save_json(json_dict, join(target_base, "dataset.json"))


def main_inverse(fold):
    # 将预测的validation label从8标签还原为14标签
    validation_dir = '/DATA5_DB8/data/sxfeng/nnUnet/nnUNet_trained_models/nnUNet/'\
                         '3d_fullres/Task101_MultiOrgan/nnUNetTrainerV2__nnUNetPlansv2.1'
    validation_raw_dir = os.path.join(validation_dir, "fold_{}/validation_raw".format(fold))
    validation_raw_inverse_dir = validation_raw_dir + '_inverse'
    maybe_mkdir_p(validation_raw_inverse_dir)

    validation_list = [i for i in os.listdir(validation_raw_dir) if 'nii.gz' in i]
    validation_list.sort()
    for p in validation_list:
        seg = join(validation_raw_dir, p)
        copy_Multiorgan_segmentation_and_convert_labels_inverse(seg, join(validation_raw_inverse_dir, p))

if __name__ == "__main__":
    main_inverse(0)