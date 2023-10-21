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
from uncertainty_refine import *
from PIL import Image

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str,
                        default="E:/Kidney/FNPC_dataset")
    parser.add_argument('--outdir', type=str,
                        default='/result1')

    parser.add_argument('--ckpt', type=str, default='sam_vit_l_0b3195.pth')
    parser.add_argument('--ref_idx', type=str, default='000') # TODO change to "000" when using kidney
    parser.add_argument('--sam_type', type=str, default='vit_l')# 'vit_h', 'vit_t', 'vit_l'

    args = parser.parse_args()
    return args

def main():
    args = get_arguments()
    print("Args:", args)

    images_path = args.data + '/Images/'
    masks_path = args.data + '/Annotations_heavy_rough/'
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

    all_dice_FP_final = []
    all_assd_FP_final = []
    all_hf_FP_final = []

    all_dice_FN_final = []
    all_assd_FN_final = []
    all_hf_FN_final = []

    for obj_name in os.listdir(images_path):
        if ".DS" not in obj_name:
            # print("object name is:", obj_name) # object name is: Subject2_20230330_135719_1
            subject_dice_original, subject_assd_original, subject_hf_original, subject_dice_ave, subject_assd_ave, \
            subject_hf_ave, subject_dice_FP_final, subject_assd_FP_final,subject_hf_FP_final, subject_dice_FN_final, \
            subject_assd_FN_final,subject_hf_FN_final = FNPC(args, obj_name, images_path, masks_path, output_path, gts_path)

            all_dice_original += subject_dice_original
            all_assd_original += subject_assd_original
            all_hf_original += subject_hf_original

            all_dice_ave += subject_dice_ave
            all_assd_ave += subject_assd_ave
            all_hf_ave += subject_hf_ave

            all_dice_FP_final += subject_dice_FP_final
            all_assd_FP_final += subject_assd_FP_final
            all_hf_FP_final += subject_hf_FP_final

            all_dice_FN_final += subject_dice_FN_final
            all_assd_FN_final += subject_assd_FN_final
            all_hf_FN_final += subject_hf_FN_final

    mean_dice_FN_final = np.mean(all_dice_FN_final)
    mean_assd_FN_final = np.mean(all_assd_FN_final )
    mean_hf_FN_final = np.mean(all_hf_FN_final)
    std_dice_FN_final = np.std(all_dice_FN_final)
    std_assd_FN_final = np.std(all_assd_FN_final)
    std_hf_FN_final = np.std(all_hf_FN_final)

    mean_dice_FP_final = np.mean(all_dice_FP_final)
    mean_assd_FP_final = np.mean(all_assd_FP_final)
    mean_hf_FP_final = np.mean(all_hf_FP_final)
    std_dice_FP_final = np.std(all_dice_FP_final)
    std_assd_FP_final = np.std(all_assd_FP_final)
    std_hf_FP_final = np.std(all_hf_FP_final)

    mean_dice_original = np.mean(all_dice_original)
    mean_assd_original = np.mean(all_assd_original)
    mean_hf_original = np.mean(all_hf_original)
    std_dice_original = np.std(all_dice_original)
    std_assd_original = np.std(all_assd_original)
    std_hf_original = np.std(all_hf_original)

    mean_dice_ave = np.mean(all_dice_ave)
    mean_assd_ave = np.mean(all_assd_ave)
    mean_hf_ave = np.mean(all_hf_ave)
    std_dice_ave = np.std(all_dice_ave)
    std_assd_ave = np.std(all_assd_ave)
    std_hf_ave = np.std(all_hf_ave)

    # uncommon the following part if you want to observe results for each data
    # all_dice_original_str = ', '.join(map(str, all_dice_original))
    # all_assd_original_str = ', '.join(map(str, all_assd_original))
    # all_hf_original_str = ', '.join(map(str, all_hf_original))
    #
    # all_dice_ave_str = ', '.join(map(str, all_dice_ave))
    # all_assd_ave_str = ', '.join(map(str, all_assd_ave))
    # all_hf_ave_str = ', '.join(map(str, all_hf_ave))
    #
    # all_dice_FP_final_str = ', '.join(map(str,all_dice_FP_final))
    # all_assd_FP_final_str = ', '.join(map(str, all_assd_FP_final))
    # all_hf_FP_final_str = ', '.join(map(str, all_hf_FP_final))
    #
    # all_dice_FN_final_str = ', '.join(map(str, all_dice_FN_final))
    # all_assd_FN_final_str = ', '.join(map(str, all_assd_FN_final))
    # all_hf_FN_final_str = ', '.join(map(str, all_hf_FN_final))


    with open(output_path+"/output.txt", "w") as file:
        # file.write(f"subject dice original: {all_dice_original_str}\n")
        # file.write(f"subject dice ave: {all_dice_ave_str}\n")
        # file.write(f"subject dice final: {all_dice_final_str}\n")
        #
        # file.write("\n")
        # file.write(f"subject assd original: {all_assd_original_str}\n")
        # file.write(f"subject assd final: {all_assd_final_str}\n")
        # file.write(f"subject assd ave: {all_assd_ave_str}\n")
        # file.write("\n")
        #
        # file.write(f"subject hf original: {all_hf_original_str}\n")
        # file.write(f"subject hf ave: {all_hf_ave_str}\n")
        # file.write(f"subject hf final: {all_hf_final_str}\n")
        # file.write("\n")

        file.write(f"mean_dice_original: {mean_dice_original}, std: {std_dice_original}\n")
        file.write(f"mean_dice_ave: {mean_dice_ave}, std: {std_dice_ave}\n")
        file.write(f"mean_dice_FN_final: {mean_dice_FN_final}, std: {std_dice_FN_final}\n")
        file.write(f"mean_dice_FNFP_final: {mean_dice_FP_final}, std: {std_dice_FP_final}\n")
        file.write("\n")
        file.write(f"mean_assd_original: {mean_assd_original}, std: {std_assd_original}\n")
        file.write(f"mean_assd_ave: {mean_assd_ave}, std: {std_assd_ave}\n")
        file.write(f"mean_assd_FN_final: {mean_assd_FN_final}, std: {std_assd_FN_final}\n")
        file.write(f"mean_assd_FNFP_final: {mean_assd_FP_final}, std: {std_assd_FP_final}\n")
        file.write("\n")
        file.write(f"mean_hf_original: {mean_hf_original}, std: {std_hf_original}\n")
        file.write(f"mean_hf_ave: {mean_hf_ave}, std: {std_hf_ave}\n")
        file.write(f"mean_hf_FN_final: {mean_hf_FN_final}, std: {std_hf_FN_final}\n")
        file.write(f"mean_hf_FNFP_final: {mean_hf_FP_final}, std: {std_hf_FP_final}\n")
        file.write("\n")


def FNPC(args, obj_name, images_path, masks_path, output_path, gts_path):
    print("\n------------> Segment " + obj_name)

    M =   # scale ratio
    N =   # number of sampled bounding box
    ave_thre = 
    uncertain_thre =  # uncertain_thre ratio

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
    sorted_files = sorted(os.listdir(test_images_path))

    print('======> Start Testing')

    subject_dice_original = []
    subject_assd_original = []
    subject_hf_original = []

    subject_dice_ave = []
    subject_assd_ave = []
    subject_hf_ave = []

    subject_dice_FN_final = []
    subject_assd_FN_final = []
    subject_hf_FN_final = []

    subject_dice_FP_final = []
    subject_assd_FP_final = []
    subject_hf_FP_final = []

    for idx in tqdm(range(len(os.listdir(test_images_path)))):
        # Load test image
        # test_idx = '%03d' % test_idx # TODO change to %03d when using kidney data
        test_idx = sorted_files[idx][:-4]#get the part except the '.png'
        # TODO change the datatype
        test_image_path = test_images_path + '/' + test_idx + '.png'
        test_image = cv2.imread(test_image_path)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

        ref_mask_path = os.path.join(masks_path, obj_name, test_idx + '.png')
        ref_mask = cv2.imread(ref_mask_path)
        ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)

        gt_mask_path = os.path.join(gts_path, obj_name, test_idx + '.png')
        gt_mask = cv2.imread(gt_mask_path)
        # print("GT mask shape:", gt_mask.shape)  # 255
        # print("GT mask max value:", np.max(gt_mask))(128, 128, 3)
        gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)
        # print("GT mask shape:", gt_mask.shape)  # 255
        # print("GT mask max value:", np.max(gt_mask))
        _, gt_mask = cv2.threshold(gt_mask, np.max(gt_mask) // 2, 1, cv2.THRESH_BINARY)
        # print("GT mask max value:", np.max(gt_mask))

        disply_original_image = test_image.copy() #
        disply_original_image2 = test_image.copy()

        # Image feature encoding
        y, x = np.nonzero(ref_mask[:,:,0])
        if y.size == 0: # in case some input doesn't have bb
            x_min = 64 - 10
            x_max = 64 + 10
            y_min = 64 - 10
            y_max = 64 + 10
        else:
            x_min = x.min()
            x_max = x.max()
            y_min = y.min()
            y_max = y.max()
        input_box = np.array([x_min, y_min, x_max, y_max])

        # ###### bbox augmentation Start #######
        new_bbox_list, radius, Center_set = sample_points2(input_box, M, N)

        # Uncommon this part to show visualize the bbox augmentation
        # save_path_demo = os.path.join(output_path, f'Demo_{test_idx}.png')
        # save_path_boxs = os.path.join(output_path, f'BoxDemo_{test_idx}.png')
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
                multimask_output=False,
                return_logits = False
            )
            mask_list.append(masks)

        # calculate the aleatoric_uncertainty
        input_mask_list = mask_list.copy()
        aleatoric_uncertainty_map = calculate_aleatoric_uncertainty(input_mask_list) # 128x128
        original_uncertainty_map = aleatoric_uncertainty_map.copy()

        ave_mask = np.mean(mask_list, axis=0)
        ave_mask = np.where(ave_mask >= ave_thre * (np.max(ave_mask)), 1, 0)

        # FNPC process:
        # For the outputs:
        # FN_UH: FN mask selected by the high uncertainty above the thre
        # FN_xUH: FN area on the image selected by the high uncertainty above the thre
        # FN_final_mask: FN correct mask. Note this mask is not refined by the bbox, so may contain some out-side-box region, but doesn't influence the result
        # FP_UH, FN_xUH: FP mask and image area selected by the high uncertainty above the thre
        # FP_final_mask: final FN_FP corrected mask, this mask will be refined by BBox before visualization

        # # if you want to use a hardcode version, please use uc_refine_hardcode
        # FN_final_mask, FN_UH, FN_xUH, FN_condition_mask, FP_final_mask, FP_UH, FP_xUH, FP_condition_mask = uc_refine_hardcode(ave_mask.copy(),
        #                                                                aleatoric_uncertainty_map.copy(),
        #                                                                test_image.copy(), threshold_uc=uncertain_thre, fn_alpha = fna,
        #                                                                fn_beta = fnb,fp_alpha = fpa, fp_beta = fpb)
        # else, use uc_refine_correct
        FN_final_mask, FN_UH, FN_xUH, FN_condition_mask, FP_final_mask, FP_UH, FP_xUH, FP_condition_mask = uc_refine_correct(ave_mask.copy(),
                                                                       aleatoric_uncertainty_map.copy(),
                                                                       test_image.copy(), threshold_uc=uncertain_thre, fn_alpha=fna,
                                                                       fn_beta=fnb, fp_alpha=fpa, fp_beta=fpb)

        # ****** Using the initial bounding box to refine the ave_mask and fine_mask ****** #
        # The prior knowledge is that the region outside the initial bounding box are all true negative
        refine_mask = np.where(ref_mask[:, :, 0] > 0, 1, 0)
        ave_mask[0] = ave_mask[0]*refine_mask
        FP_final_mask[0] = FP_final_mask[0]*refine_mask

        # print(FP_final_mask.shape)

        print("the max number of FP_xUH is:", np.max(FP_xUH))
        print("the min number of FP_xUH is:", np.min(FP_xUH))

        # the mask_list is in true or false farmat, turn it into 0, 1
        original_mask = mask_list[0][0].copy()
        original_mask = original_mask.astype(int)

        # calculate Dice, ASSD, hausdorff
        case_dice_original = dice_score(gt_mask, original_mask)
        case_assd_original = assd(gt_mask, original_mask)
        case_hf_original = hausdorff_distance(gt_mask, original_mask)

        case_dice_ave = dice_score(gt_mask, ave_mask[0])
        case_assd_ave = assd(gt_mask, ave_mask[0])
        case_hf_ave = hausdorff_distance(gt_mask, ave_mask[0])

        # final_mask = FP_final_mask.copy()
        print("FP_final_mask.copy().shape",FP_final_mask.copy().shape)
        print("FP_final_mask.copy() max", np.max(FP_final_mask.copy()))
        case_dice_FP_final = dice_score(gt_mask, FP_final_mask[0])
        case_assd_FP_final = assd(gt_mask, FP_final_mask[0])
        case_hf_FP_final = hausdorff_distance(gt_mask, FP_final_mask[0])

        case_dice_FN_final = dice_score(gt_mask, FN_final_mask[0])
        case_assd_FN_final = assd(gt_mask, FN_final_mask[0])
        case_hf_FN_final = hausdorff_distance(gt_mask, FN_final_mask[0])

        subject_dice_original.append(case_dice_original)
        subject_assd_original.append(case_assd_original)
        subject_hf_original.append(case_hf_original)

        subject_dice_ave.append(case_dice_ave)
        subject_assd_ave.append(case_assd_ave)
        subject_hf_ave.append(case_hf_ave)

        subject_dice_FN_final.append(case_dice_FN_final)
        subject_assd_FN_final.append(case_assd_FN_final)
        subject_hf_FN_final.append(case_hf_FN_final)

        subject_dice_FP_final.append(case_dice_FP_final)
        subject_assd_FP_final.append(case_assd_FP_final)
        subject_hf_FP_final.append(case_hf_FP_final)

        # visualization the FNPC process

        fig, axs = plt.subplots(3, 4, figsize=(10 * 3, 10 * 4))

        axs[0 , 0].imshow(test_image)
        show_mask(original_mask, axs[0 , 0])
        show_box(new_bbox_list[0], axs[0 , 0], color="yellow")
        axs[0 , 0].set_title(f'Ori_D_{case_dice_original:.2f}_A_{case_assd_original:.2f}_H_{case_hf_original:.2f}',
                         fontsize=18)

        axs[0 ,1].imshow(test_image)
        show_mask(ave_mask, axs[0 ,1])
        show_box(new_bbox_list[0], axs[0 ,1], color="yellow")
        axs[0 ,1].set_title(f'Ave_D_{case_dice_ave:.2f}_A_{case_assd_ave:.2f}_H_{case_hf_ave:.2f}', fontsize=18)

        # 可视化uncertainty map
        vis_img = test_image.copy()
        # axs[4].imshow(test_image)
        # show_uncertainty(aleatoric_uncertainty_map, plt.gca())
        overlay_uncertainty_on_image(vis_img, original_uncertainty_map, axs[0, 2])
        axs[0, 2].set_title(f"Uncertainty Map", fontsize=18)

        axs[0 ,3].imshow(test_image)
        show_mask(np.expand_dims(gt_mask, axis=0), axs[0 ,3])
        show_box(new_bbox_list[0], axs[0 ,3], color="yellow")
        axs[0 ,3].set_title(f'GT', fontsize=18)

        # FN UH
        axs[1, 0].imshow(FN_UH)
        axs[1, 0].set_title(f"FN UH", fontsize=18)

        # # FN xUH
        axs[1, 1].imshow(FN_xUH, cmap='gray', vmin=0, vmax=255)
        axs[1, 1].set_title(f"FN xUH", fontsize=18)

        # # FN condition
        axs[1, 2].imshow(FN_condition_mask)
        axs[1, 2].set_title(f"FN_condition_mask", fontsize=18)

        axs[1, 3].imshow(test_image)
        show_mask(FN_final_mask, axs[1, 3])
        show_box(new_bbox_list[0], axs[1, 3], color="yellow")
        axs[1, 3].set_title(f'FN_D_{case_dice_FN_final:.2f}_A_{case_assd_FN_final:.2f}_H_{case_hf_FN_final:.2f}', fontsize=18)

        # FP UH
        axs[2, 0].imshow(FP_UH)
        axs[2, 0].set_title(f"FP UH", fontsize=18)

        # # FP xUH
        axs[2, 1].imshow(FP_xUH, cmap='gray', vmin=0, vmax=255)
        axs[2, 1].set_title(f"FP xUH", fontsize=18)

        # # FN condition
        axs[2, 2].imshow(FP_condition_mask)
        axs[2, 2].set_title(f"FP_condition_mask", fontsize=18)

        axs[2, 3].imshow(test_image)
        show_mask(FP_final_mask, axs[2, 3])
        show_box(new_bbox_list[0], axs[2, 3], color="yellow")
        axs[2, 3].set_title(f'FP_FN_D_{case_dice_FP_final:.2f}_A_{case_assd_FP_final:.2f}_H_{case_hf_FP_final:.2f}', fontsize=18)


        # Save the subplot panel as a single image
        vis_mask_output_path = os.path.join(output_path, f'Results_Comparison_{test_idx}.png')
        plt.savefig(vis_mask_output_path, format='png')


        ######### here save all the prediction masks for the reconstruction
        # save all the predicted mask for the next loop:
        ori_save_name = os.path.join(output_path, "ori_fake_mask_" + str(test_idx) + ".png")  # since it's from 0
        ave_save_name = os.path.join(output_path, "ave_fake_mask_" + str(test_idx) + ".png")
        final_save_name = os.path.join(output_path, "final_fake_mask_" + str(test_idx) + ".png")

        slice_y = (mask_list[0][0].copy().astype(np.float32) - np.min(mask_list[0][0].copy().astype(np.float32))) / (
                    np.max(mask_list[0][0].copy().astype(np.float32)) -
                    np.min(mask_list[0][
                               0].copy().astype(np.float32)) + 1e-10) * 255  # normalize to 0-255
        slice_y = slice_y.astype(np.uint8)
        Image.fromarray(slice_y).save(ori_save_name)  # save with PIL instead of plt

        slice_y = (ave_mask[0].copy().astype(np.float32) - np.min(ave_mask[0].copy().astype(np.float32))) / (
                    np.max(ave_mask[0].copy().astype(np.float32)) -
                    np.min(ave_mask[
                               0].copy().astype(np.float32)) + 1e-10) * 255  # normalize to 0-255
        slice_y = slice_y.astype(np.uint8)
        Image.fromarray(slice_y).save(ave_save_name)  # save with PIL instead of plt

        slice_y = (FP_final_mask[0].copy().astype(np.float32) - np.min(FP_final_mask[0].copy().astype(np.float32))) / (
                    np.max(FP_final_mask[0].copy().astype(np.float32)) -
                    np.min(FP_final_mask[
                               0].copy().astype(np.float32)) + 1e-10) * 255  # normalize to 0-255
        slice_y = slice_y.astype(np.uint8)
        Image.fromarray(slice_y).save(final_save_name)  # save with PIL instead of plt


    return subject_dice_original, subject_assd_original, subject_hf_original, subject_dice_ave, subject_assd_ave, \
           subject_hf_ave, subject_dice_FP_final, subject_assd_FP_final,subject_hf_FP_final, subject_dice_FN_final, subject_assd_FN_final,subject_hf_FN_final


if __name__ == "__main__":
    main()
