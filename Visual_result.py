import matplotlib.pyplot as plt
import numpy as np
import torch
from unet import UNet
import SimpleITK as sitk
import os
import glob
from torchvision import transforms as T
from torch.utils.data import TensorDataset
import argparse
from PIL import Image


def volume_resample_crop(image, spacing, crop_size, image_name):
    # image: input simpleitk image list
    # spacing: desired(output) spacing
    # crop_size: desired(output) image size
    # image_name: could be one of "T2w", "Adc", "Hbv", "Lesion" or "Prostate"

    orig_size = np.array(image.GetSize())
    new_spacing = np.array(spacing)
    orig_spacing = image.GetSpacing()
    new_size = orig_size * (orig_spacing / new_spacing)
    new_size = np.floor(new_size)
    new_size = [int(s) for s in new_size]
    pad_value = image.GetPixelIDValue()
    # Create the ResampleImageFilter
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(list(new_spacing))
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(pad_value)
    if image_name == "Lesion" or image_name == "Prostate":
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkBSpline)
    # Execute the filter
    resample_image = resampler.Execute(image)
    # Get the size of the MRI volume
    size = resampler.GetSize()
    # Get the center voxel index of the MRI volume
    center = [int(size[0] / 2), int(size[1] / 2), int(size[2] / 2)]
    # Calculate the start and end indices for each dimension
    start_index = [center[i] - int(crop_size[i] / 2) for i in range(3)]
    end_index = [start_index[i] + crop_size[i] for i in range(3)]
    cropper = sitk.CropImageFilter()
    # Set the crop boundaries
    cropper.SetLowerBoundaryCropSize(start_index)
    cropper.SetUpperBoundaryCropSize([size[i] - end_index[i] for i in range(3)])
    # Crop the volume
    resample_cropped_volume = cropper.Execute(resample_image)

    return resample_cropped_volume


def slice_data(image, image_name, fold_number, new_spacing, crop_size):
    if image_name == "T2w":
        var_name = "image_T2w_list_Fold_" + str(fold_number)
        dict_vars = {}
        dict_vars[var_name] = []
        resample_image_T2w = volume_resample_crop(image, new_spacing, crop_size, image_name)
        for z in range(resample_image_T2w.GetSize()[2]):
            z_slice = sitk.Extract(resample_image_T2w, [resample_image_T2w.GetSize()[0],
                                                        resample_image_T2w.GetSize()[1], 0],
                                   [0, 0, z])
            dict_vars[var_name].append(z_slice)
    elif image_name == "Adc":
        var_name = "image_Adc_list_Fold_" + str(fold_number)
        dict_vars = {}
        dict_vars[var_name] = []
        resample_image_Adc = volume_resample_crop(image, new_spacing, crop_size, image_name)
        for z in range(resample_image_Adc.GetSize()[2]):
            z_slice = sitk.Extract(resample_image_Adc, [resample_image_Adc.GetSize()[0],
                                                        resample_image_Adc.GetSize()[1], 0],
                                   [0, 0, z])
            dict_vars[var_name].append(z_slice)
    elif image_name == "Hbv":
        var_name = "image_Hbv_list_Fold_" + str(fold_number)
        dict_vars = {}
        dict_vars[var_name] = []
        resample_image_Hbv = volume_resample_crop(image, new_spacing, crop_size, image_name)
        for z in range(resample_image_Hbv.GetSize()[2]):
            z_slice = sitk.Extract(resample_image_Hbv, [resample_image_Hbv.GetSize()[0],
                                                        resample_image_Hbv.GetSize()[1], 0],
                                   [0, 0, z])
            dict_vars[var_name].append(z_slice)
    elif image_name == "Lesion":
        var_name = "image_Lesion_list_Fold_" + str(fold_number)
        dict_vars = {}
        dict_vars[var_name] = []
        resample_image_Lesion = volume_resample_crop(image, new_spacing, crop_size, image_name)
        for z in range(resample_image_Lesion.GetSize()[2]):
            z_slice = sitk.Extract(resample_image_Lesion, [resample_image_Lesion.GetSize()[0],
                                                           resample_image_Lesion.GetSize()[1], 0],
                                   [0, 0, z])
            dict_vars[var_name].append(z_slice)
    elif image_name == "Prostate":
        var_name = "image_Prostate_list_Fold_" + str(fold_number)
        dict_vars = {}
        dict_vars[var_name] = []
        resample_image_Prostate = volume_resample_crop(image, new_spacing, crop_size, image_name)
        for z in range(resample_image_Prostate.GetSize()[2]):
            z_slice = sitk.Extract(resample_image_Prostate, [resample_image_Prostate.GetSize()[0],
                                                             resample_image_Prostate.GetSize()[1], 0],
                                   [0, 0, z])
            dict_vars[var_name].append(z_slice)

    return dict_vars[var_name]


# This function performs augmentation and normalization for validation and test sets (written by Moslem)
def augment_data(img_list, data_mean, data_std, data_type):
    if data_type == "data":
        transform_tr = T.Compose([
            T.Normalize(data_mean, data_std)
        ])
        transformed_imgs = []
        for img in img_list:
            float_tensor = T.ToTensor()(Image.fromarray(img)).float()
            img_tensor = transform_tr(float_tensor)
            transformed_imgs.append(img_tensor)

    elif data_type == "target":
        transformed_imgs = []
        for img in img_list:
            float_tensor = T.ToTensor()(Image.fromarray(img)).float()
            transformed_imgs.append(float_tensor)
    return transformed_imgs


def visualize(net_t2w, net_adc, net_hbv, data_t2w, data_adc, data_hbv):
    # Visualization for T2w sequences
    for i, (inputs, labels) in enumerate(data_t2w):
        inputs, labels = inputs.to(device), labels.to(device)
        mask_true = labels
        with torch.no_grad():
            # predict the mask
            mask_pred = net_t2w(inputs)
            plt.figure(1)
            plt.imshow(inputs.squeeze().cpu().numpy(), cmap=plt.cm.gray)
            plt.title('Original 2D MRI slice')
            plt.show()
            # Plot of the original mask
            plt.figure(2)
            plt.imshow(mask_true.squeeze().cpu().numpy(), cmap=plt.cm.Greens)
            plt.title('Original mask', fontsize=16)
            plt.show()
            # pre-process mask prediction
            mask_pred = mask_pred.squeeze().cpu().numpy()
            mask_pred = convert_pred_to_image(mask_pred, mask_true)

            # Plot of the predicted mask
            plt.figure(3)
            plt.imshow(mask_pred.astype(np.float32), cmap=plt.cm.Reds)
            plt.title('Predicted mask', fontsize=16)
            plt.show()
            # Plot of the overlay
            plt.figure(4)
            plt.imshow(mask_pred, cmap=plt.cm.Reds, alpha=0.5)
            plt.imshow(mask_true.squeeze().cpu().numpy(), cmap=plt.cm.gray, alpha=0.5)
            plt.title('Plot of the overlay two masks', fontsize=16)
            plt.show()

    # Visualization for Adc sequences
    for i, (inputs, labels) in enumerate(data_adc):
        inputs, labels = inputs.to(device), labels.to(device)
        mask_true = labels

        with torch.no_grad():
            # predict the mask
            mask_pred = net_adc(inputs)
            plt.figure(1)
            plt.imshow(inputs.squeeze().cpu().numpy(), cmap=plt.cm.gray)
            plt.title('Original 2D MRI slice', fontsize=16)
            plt.show()
            # Plot of the original mask
            plt.figure(2)
            plt.imshow(mask_true.squeeze().cpu().numpy(), cmap=plt.cm.Greens)
            plt.title('Original mask', fontsize=16)
            plt.show()
            # pre-process mask prediction for visualization
            mask_pred = mask_pred.squeeze().cpu().numpy()
            mask_pred = convert_pred_to_image(mask_pred, mask_true)

            # Plot of the predicted mask
            plt.figure(3)
            plt.imshow(mask_pred, cmap=plt.cm.Reds)
            plt.title('Predicted mask', fontsize=16)
            plt.show()
            # Plot of the overlay
            plt.figure(4)
            plt.imshow(mask_pred, cmap=plt.cm.Reds, alpha=0.5)
            plt.imshow(mask_true.squeeze().cpu().numpy(), cmap=plt.cm.gray, alpha=0.5)
            plt.title('Plot of the overlay of two images', fontsize=16)
            plt.show()

    # Visualization for Hbv sequences
    for i, (inputs, labels) in enumerate(data_hbv):
        inputs, labels = inputs.to(device), labels.to(device)
        mask_true = labels

        with torch.no_grad():
            # predict the mask
            mask_pred = net_hbv(inputs)
            plt.figure(1)
            plt.imshow(inputs.squeeze().cpu().numpy(), cmap=plt.cm.gray)
            plt.title('Original 2D MRI slice of ', fontsize=16)
            plt.show()
            # Plot of the original mask
            plt.figure(2)
            plt.imshow(mask_true.squeeze().cpu().numpy(), cmap=plt.cm.Greens)
            plt.title('Original mask for the MRI slice of ', fontsize=16)
            plt.show()
            # pre-process mask prediction
            mask_pred = mask_pred.squeeze().cpu().numpy()
            mask_pred = convert_pred_to_image(mask_pred, mask_true)

            # Plot of the predicted mask
            plt.figure(3)
            plt.imshow(mask_pred, cmap=plt.cm.Reds)
            plt.title('Plot of the predicted mask', fontsize=16)
            plt.show()
            # Plot of the overlay of
            plt.figure(4)
            plt.imshow(mask_pred, cmap=plt.cm.Reds, alpha=0.5)
            plt.imshow(mask_true.squeeze().cpu().numpy(), cmap=plt.cm.gray, alpha=0.5)
            plt.title('Plot of the overlay of two images', fontsize=16)
            plt.show()


def convert_pred_to_image(data, mask):
    # function to convert the one-hot output to image mask
    data_output = np.ones((data.shape[1], data.shape[2]))
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            # find the prediction result for each pixel
            if data[0, i, j] >= abs(data[1, i, j]):
                data_output[i, j] = 1
            else:
                data_output[i, j] = 0

    return data_output * mask.squeeze().cpu().numpy()


def remove_dim_one(image_in):
    size = image_in.GetSize()
    # If the fourth dimension has a size of 1, extract the 3D volume
    # If the last dimension has a size of 1, extract the 3D volume
    if size[-1] == 1 and len(size):
        # Make sure that the last dimension is not zero before extraction
        if size[-2] > 0:
            # Make sure the image has at least 3 dimensions
            if image_in.GetDimension() >= 3:
                # Make sure the extraction indices are within the bounds of the image
                if size[-2] > 0 and size[-1] > 0:
                    # Extract the 3D volume without the last dimension
                    image2 = sitk.Extract(image_in,
                                          [image_in.GetSize()[0], image_in.GetSize()[1], image_in.GetSize()[2], 0],
                                          [0, 0, 0, 0])
                else:
                    print("The size of the last dimension is zero. Cannot extract.")
            else:
                print("Image has fewer than 3 dimensions. Cannot extract.")
        else:
            print("The size of the last dimension is zero. Cannot extract.")
    else:
        image2 = image_in
    return image2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Assignment 3')
    # hyper-parameters
    parser.add_argument('-BatchSize', default=1, type=int, metavar='N', help='mini-batch size Default: 8')
    parser.add_argument('-data_path', default='D:/Courses/CISC_881_Medical_Imaging/PICAI_dataset/', help='path to data')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    args = parser.parse_args()

    data_filenames = glob.glob(os.path.join(args.data_path, "picai_public_images_fold0/10522/*.mha"))
    prostate_filenames = glob.glob(os.path.join(args.data_path, "picai_labels-main/anatomical_delineations"
                                                                "/whole_gland/AI/Bosma22b/10522_1000532.nii"))
    lesion_filenames = glob.glob(os.path.join(args.data_path, "picai_labels-main/csPCa_lesion_delineations/AI/Bosma22a"
                                                              "/10522_1000532.nii"))

    # set the size and spacing for all data
    output_spacing = [0.5, 0.5, 3.0]
    output_size = [300, 300, 16]
    mean_T2w = 209.71
    std_T2w = 134.86
    mean_Adc = 777.54
    std_Adc = 701.82
    mean_Hbv = 11.81
    std_Hbv = 7.65

    # visualization function according to the requirement of assignment 3
    Images = [sitk.ReadImage(filename) for filename in data_filenames]
    prostate_mask = sitk.ReadImage(prostate_filenames)
    lesion_mask = sitk.ReadImage(lesion_filenames)
    Images_Adc = Images[0]  # Read the adc input image
    Images_Hbv = Images[2]  # Read the hbv input image
    Images_T2w = Images[4]  # Read the t2w input image

    Images_Adc = remove_dim_one(Images_Adc)
    Images_Hbv = remove_dim_one(Images_Hbv)
    Images_Adc = remove_dim_one(Images_Adc)
    prostate_mask = remove_dim_one(prostate_mask)
    lesion_mask = remove_dim_one(lesion_mask)

    # apply the desirable size and spacing with slicing the z dimension
    Images_T2w = slice_data(Images_T2w, "T2w", 0, output_spacing, output_size)
    prostate_mask = slice_data(prostate_mask, "Prostate", 0, output_spacing, output_size)
    Images_Adc = slice_data(Images_Adc, "Adc", 0, output_spacing, output_size)
    Images_Hbv = slice_data(Images_Hbv, "Hbv", 0, output_spacing, output_size)
    lesion_mask = slice_data(lesion_mask, "Lesion", 0, output_spacing, output_size)

    np_Images_Adc = []
    np_Images_T2w = []
    np_Images_Hbv = []
    np_prostate_mask = []
    np_lesion_mask = []
    for sitk_image in Images_Adc:
        np_imagea = sitk.GetArrayFromImage(sitk_image)
        np_Images_Adc.append(np_imagea)

    for sitk_image in Images_T2w:
        np_imaget = sitk.GetArrayFromImage(sitk_image)
        np_Images_T2w.append(np_imaget)

    for sitk_image in Images_Hbv:
        np_imageh = sitk.GetArrayFromImage(sitk_image)
        np_Images_Hbv.append(np_imageh)

    for sitk_image in prostate_mask:
        np_imagep = sitk.GetArrayFromImage(sitk_image)
        np_prostate_mask.append(np_imagep)

    for sitk_image in lesion_mask:
        np_imagel = sitk.GetArrayFromImage(sitk_image)
        np_lesion_mask.append(np_imagel)

    # Apply the transforms (augmentation and normalization)
    transformed_test_imgs_T2w = augment_data(np_Images_T2w, mean_T2w, std_T2w, "data")
    transformed_test_targets_T2w = augment_data(np_prostate_mask, mean_T2w, std_T2w, "target")
    transformed_test_imgs_Adc = augment_data(np_Images_Adc, mean_Adc, std_Adc, "data")
    transformed_test_targets_Adc_Hbv = augment_data(np_lesion_mask, mean_Adc, std_Adc, "target")
    transformed_test_imgs_Hbv = augment_data(np_Images_Hbv, mean_Hbv, std_Hbv, "data")

    # prepare data for the Unet
    test_dataset_Tw2 = TensorDataset(torch.stack(transformed_test_imgs_T2w), torch.stack(transformed_test_targets_T2w))
    test_loader_T2w = torch.utils.data.DataLoader(test_dataset_Tw2, batch_size=args.BatchSize)
    test_dataset_Adc = TensorDataset(torch.stack(transformed_test_imgs_Adc),
                                     torch.stack(transformed_test_targets_Adc_Hbv))
    test_loader_Adc = torch.utils.data.DataLoader(test_dataset_Adc, batch_size=args.BatchSize)
    test_dataset_Hbv = TensorDataset(torch.stack(transformed_test_imgs_Hbv),
                                     torch.stack(transformed_test_targets_Adc_Hbv))
    test_loader_Hbv = torch.utils.data.DataLoader(test_dataset_Hbv, batch_size=args.BatchSize)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_t2w = UNet(n_channels=1, n_classes=2)
    model_t2w.to(device)
    model_t2w.load_state_dict(torch.load('./checkpoints/T2w/checkpoint_epoch_11.pth', map_location=device))

    model_Adc = UNet(n_channels=1, n_classes=2)
    model_Adc.to(device)
    model_Adc.load_state_dict(torch.load('./checkpoints/Adc/checkpoint_epoch_11.pth', map_location=device))

    model_Hbv = UNet(n_channels=1, n_classes=2)
    model_Hbv.to(device)
    model_Hbv.load_state_dict(torch.load('./checkpoints/Hbv/checkpoint_epoch_11.pth', map_location=device))

    visualize(model_t2w, model_Adc, model_Hbv, test_loader_T2w, test_loader_Adc, test_loader_Hbv)
