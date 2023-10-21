import numpy as np
import torch
from torch.nn import functional as F

import os
import cv2
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

from show import *
from segment_anything import sam_model_registry, SamPredictor

from itertools import combinations

from scipy.spatial import ConvexHull
import numpy as np
import itertools
import scipy
from scipy.spatial.distance import cdist
from prompt_aug import *
from loss_aug_sam import *
from natsort import natsorted
from PIL import Image
from uncertainty_refine import *
# follows 2D_to_3D_aug5_2 but utilize the bounding box refine

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str,
                        default="/dataset/Placenta/manualSeg/for_SAM/Reconstruction")
    parser.add_argument('--outdir', type=str,
                        default='Plancenta/Reconstruction/subjectx')

    parser.add_argument('--ckpt', type=str, default='sam_vit_l_0b3195.pth')
    parser.add_argument('--ref_idx', type=str, default='000') # TODO change to "000" when using kidney
    parser.add_argument('--sam_type', type=str, default='vit_l')# 'vit_h', 'vit_t', 'vit_l'

    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    print("Args:", args)

    images_path = args.data + '/Images_Subjectx/'
    masks_path = args.data + '/Annotations_0002/'
    gts_path = args.data + '/GTs/'
    output_path = './outputs/' + args.outdir

    if not os.path.exists('./outputs/'):
        os.mkdir('./outputs/')

    all_dice_original = []
    all_assd_original = []
    all_hf_original = []

    all_dice_ave = []
    all_assd_ave = []
    all_hf_ave = []

    all_dice_final = []
    all_assd_final = []
    all_hf_final = []

    for obj_name in os.listdir(images_path):
        if ".DS" not in obj_name:
            # print("object name is:", obj_name) # object name is: Subject2_20230330_135719_1
            subject_dice_original, subject_assd_original, subject_hf_original, subject_dice_ave, subject_assd_ave, \
            subject_hf_ave, subject_dice_final, subject_assd_final,subject_hf_final = SS2V(args, obj_name, images_path, masks_path, output_path, gts_path)

def SS2V(args, obj_name, images_path, masks_path, output_path, gts_path):
    print("\n------------> Segment " + obj_name)
    M =   # scale ratio
    N =   # number of sampled bounding box
    ave_thre = 
    uncertain_thre =   # uncertain_thre ratio

    # for kidney the value of FN should be small, which means the range should be small
    #
    fna =   # include more FN which outside the avemask but inside the UM, with value fna< and < fnb
    fnb = 
    fpa =   # exclude FP which inside the avemask and UM, with value<fpa, or value>fpb
    fpb = 

    # Path preparation
    test_images_path = os.path.join(images_path, obj_name)

    output_path = os.path.join(output_path, obj_name)
    os.makedirs(output_path, exist_ok=True)

    # Load images and masks
    # ref_image = cv2.imread(ref_image_path)
    # ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
    #
    # ref_mask = cv2.imread(ref_mask_path)
    # ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)

    print("======> Load SAM")
    if args.sam_type == 'vit_h':
        # TODO add a weights/
        # sam_type, sam_ckpt = 'vit_h', 'weights/sam_vit_h_4b8939.pth'
        sam_type, sam_ckpt = 'vit_h', 'weights/sam_vit_h_4b8939.pth'
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).cuda()
        sam.eval()
    elif args.sam_type == 'vit_t':
        sam_type, sam_ckpt = 'vit_t', 'weights/mobile_sam.pt'
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).to(device=device)
        sam.eval()

    # here add more sam archs
    elif args.sam_type == 'vit_l':
        sam_type, sam_ckpt = 'vit_l', 'weights/sam_vit_l_0b3195.pth'
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).to(device=device)
        sam.eval()

    predictor = SamPredictor(sam)
    # print(os.listdir(test_images_path))
    # print(test_images_path)
    sorted_files = natsorted(os.listdir(test_images_path))
    # print(sorted_files)

    print('======> Start Testing')

    subject_dice_original = []
    subject_assd_original = []
    subject_hf_original = []

    subject_dice_ave = []
    subject_assd_ave = []
    subject_hf_ave = []

    subject_dice_final = []
    subject_assd_final = []
    subject_hf_final = []

    # the start slice number and the end slice number
    start_num = 59
    end_num = 94
    minx = 2
    miny = 2
    maxx = 2
    maxy = 2
    for idx in tqdm(range(start_num, end_num+1)):
        print(idx)
        # Load test image
        # TODO change to %03d when using kidney data
        test_idx = sorted_files[idx][:-4]#get the part except the '.png'
        print("*********************",test_idx)
        # TODO change the datatype
        # test_image_path = test_images_path + '/' + test_idx + '.jpg'
        test_image_path = test_images_path + '/' + test_idx + '.png'
        # print("test image path:",test_image_path)
        test_image = cv2.imread(test_image_path)
        print("test_image shape:", test_image.shape)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

        # if the input image is the middle slice, extract the reference mask
        if idx == start_num:
            ref_mask_path = os.path.join(masks_path, obj_name, "slice_" +str(start_num)+".png")
            print(ref_mask_path)
            ref_mask = cv2.imread(ref_mask_path)
            ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)
        else: # if other slices, get the bounding box from the previous slice
            ref_mask_path = os.path.join(output_path, "final_fake_box_" + str(idx - 1) + ".png")# get the fake_box generate from the previous one
            ref_mask = cv2.imread(ref_mask_path)
            ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)


        gt_mask_path = os.path.join(gts_path, obj_name, test_idx + '.png')
        gt_mask = cv2.imread(gt_mask_path)
        gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)
        _, gt_mask = cv2.threshold(gt_mask, np.max(gt_mask) // 2, 1, cv2.THRESH_BINARY)

        # Image feature encoding
        y, x = np.nonzero(ref_mask[:,:,0])
        if y.size == 0:
            x_min = 0
            x_max = 128
            y_min = 0
            y_max = 128
        else:
            x_min = x.min()
            x_max = x.max()
            y_min = y.min()
            y_max = y.max()

        input_box = np.array([x_min, y_min, x_max, y_max])

        # ###### bbox augmentation Start #######
        new_bbox_list, radius, Center_set = sample_points2(input_box, M, N)

        save_path_demo = os.path.join(output_path, f'Demo_{test_idx}.png')
        save_path_boxs = os.path.join(output_path, f'BoxDemo_{test_idx}.png')
        # visualize_and_save(disply_original_image, new_bbox_list, VT1_set, VT2_set, VT3_set, VT4_set, radius, save_path_demo, save_path_boxs)

        # generate mask for bbox list
        mask_list = []
        for i in range(len(new_bbox_list)):
            predictor.set_image(test_image)
            # print("the box i is:", new_bbox_list[i][None, :])
            masks, scores, logits, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=new_bbox_list[i][None, :],
                multimask_output=False
            )
            # print('masks shape:', masks.shape) # (1, 128, 128)
            mask_list.append(masks)
        best_idx = 0

        # calculate the aleatoric_uncertainty
        # 计算aleatoric uncertainty
        input_mask_list = mask_list.copy()
        aleatoric_uncertainty_map = calculate_aleatoric_uncertainty(input_mask_list) # 128x128

        ave_mask = np.mean(mask_list, axis=0)
        ave_mask = np.where(ave_mask >= ave_thre * (np.max(ave_mask)), 1, 0)

        # calculate the ave_mask
        # final_mask = simple_threshold(aleatoric_uncertainty_map, uncertainty_threshold = 0.5)
        FN_final_mask, FN_UH, FN_xUH, FN_condition_mask, FP_final_mask, FP_UH, FP_xUH, FP_condition_mask = uc_refine_hardcode(
            ave_mask.copy(),
            aleatoric_uncertainty_map.copy(),
            test_image.copy(), threshold_uc=uncertain_thre, fn_alpha=fna,
            fn_beta=fnb, fp_alpha=fpa, fp_beta=fpb)

        # using the initial bounding box to refine the ave_mask and fine_mask
        refine_mask = np.where(ref_mask[:, :, 0] > 0, 1, 0)
        ave_mask[0] = ave_mask[0] * refine_mask
        FP_final_mask[0] = FP_final_mask[0] * refine_mask

        ### **** generate new mask for the next image **** ####
        y_new, x_new = np.nonzero(FP_final_mask[0].copy())
        if y_new.size == 0:
            x_new_min = 0
            x_new_max = 128
            y_new_min = 0
            y_new_max = 128
        else:
            x_new_min = x_new.min()
            x_new_max = x_new.max()
            y_new_min = y_new.min()
            y_new_max = y_new.max()


        # compare the new mask with old mask to correct the position
        if np.abs(x_new_min - x_min) > minx:
            x_new_min = x_min.copy()
        if np.abs(y_new_min - y_min) > miny:
            y_new_min = y_min.copy()
        if np.abs(x_new_max - x_max) > maxx:
            x_new_max = x_max.copy()
        if np.abs(y_new_max - y_max) > maxy:
            y_new_max = y_max.copy()

        # generate the fake mask for the next input
        new_box = np.array([x_new_min, y_new_min, x_new_max, y_new_max])
        new_binary_mask = create_mask_from_bbox(new_box, size=128)

        new_binary_mask_name = os.path.join(output_path,"final_fake_box_" + str(idx) + ".png")
        Image.fromarray(new_binary_mask).save(new_binary_mask_name)  # save with PIL instead of plt

        # the mask_list is in true or false farmat, turn it into 0, 1
        original_mask = mask_list[0][0].copy()
        original_mask = original_mask.astype(int)



        # compare results we get
        fig, axs = plt.subplots(1, 5, figsize=(10 * 5, 10))

        axs[0].imshow(test_image)
        show_mask(original_mask, axs[0])
        show_box(new_bbox_list[0], axs[0], color="yellow")
        # axs[0].set_title(f'Ori_D_{case_dice_original:.2f}_A_{case_assd_original:.2f}_H_{case_hf_original:.2f}', fontsize=18)
        axs[0].set_title(f'Ori',
                         fontsize=18)

        axs[1].imshow(test_image)
        show_mask(ave_mask, axs[1])
        show_box(new_bbox_list[0], axs[1], color="yellow")
        axs[1].set_title(f'Ave', fontsize=18)

        axs[2].imshow(test_image)
        show_mask(FP_final_mask, axs[2])
        show_box(new_bbox_list[0], axs[2], color="yellow")
        axs[2].set_title(f'Fin', fontsize=18)

        axs[3].imshow(test_image)
        show_mask(np.expand_dims(gt_mask, axis=0), axs[3])
        show_box(new_bbox_list[0], axs[3], color="yellow")
        axs[3].set_title(f'GT', fontsize=18)

        # visualize uncertainty map
        vis_img = test_image.copy()
        overlay_uncertainty_on_image(vis_img, aleatoric_uncertainty_map.copy(), axs[4])
        axs[4].set_title(f"Uncertainty Map", fontsize=18)

        # Save the subplot panel as a single image
        vis_mask_output_path = os.path.join(output_path, f'Results_Comparison_{test_idx}.png')
        plt.savefig(vis_mask_output_path, format='png')

        # save all the predicted mask for the next loop:
        ori_save_name = os.path.join(output_path, "ori_fake_mask_" + str(idx) + ".png") # since it's from 0
        ave_save_name = os.path.join(output_path, "ave_fake_mask_" + str(idx) + ".png")
        final_save_name = os.path.join(output_path,"final_fake_mask_" + str(idx) + ".png")

        slice_y = (mask_list[0][0].copy().astype(np.float32) - np.min(mask_list[0][0].copy().astype(np.float32))) / (np.max(mask_list[0][0].copy().astype(np.float32)) -
                                                                               np.min(mask_list[0][
                                                                                          0].copy().astype(np.float32)) + 1e-10) * 255  # normalize to 0-255
        slice_y = slice_y.astype(np.uint8)
        Image.fromarray(slice_y).save(ori_save_name)  # save with PIL instead of plt

        slice_y = (ave_mask[0].copy().astype(np.float32) - np.min(ave_mask[0].copy().astype(np.float32))) / (np.max(ave_mask[0].copy().astype(np.float32)) -
                                                                       np.min(ave_mask[
                                                                                  0].copy().astype(np.float32)) + 1e-10) * 255  # normalize to 0-255
        slice_y = slice_y.astype(np.uint8)
        Image.fromarray(slice_y).save(ave_save_name)  # save with PIL instead of plt

        slice_y = (FP_final_mask[0].copy().astype(np.float32) - np.min(FP_final_mask[0].copy().astype(np.float32))) / (np.max(FP_final_mask[0].copy().astype(np.float32)) -
                                                                           np.min(FP_final_mask[
                                                                                      0].copy().astype(np.float32)) + 1e-10) * 255  # normalize to 0-255
        slice_y = slice_y.astype(np.uint8)
        Image.fromarray(slice_y).save(final_save_name)  # save with PIL instead of plt

    return subject_dice_original, subject_assd_original, subject_hf_original, subject_dice_ave, subject_assd_ave, \
           subject_hf_ave, subject_dice_final, subject_assd_final,subject_hf_final

if __name__ == "__main__":
    main()
