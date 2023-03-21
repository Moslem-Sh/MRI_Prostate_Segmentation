import SimpleITK as sitk
import os
import glob
import numpy as np
from torchvision import transforms as T
import torch
from PIL import Image
from torch.utils.data import TensorDataset
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from unet import UNet
import errno
from tqdm import tqdm
import argparse
from evaluate import evaluate
from utils.dice_score import dice_loss


class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0.0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_losse, validation_losse):
        if (validation_losse - train_losse) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True


# This function performs corresponding fold reading of the target folders (written by Moslem)
def read_folds(Filenames_Images, Filenames_Lesion_Mask):
    Fold_Lesion_Mask_pathes = []
    # loop through each directory
    for directory_path in Filenames_Images:
        # get the base name of the file
        file_base = os.path.basename(directory_path)
        # split the base name into its name and extension parts
        file_name, file_extension = os.path.splitext(file_base)
        # extract the desired part from the file name
        desired_part1 = file_name.split("_")[0]
        desired_part2 = file_name.split("_")[1]
        desired_part = str(Filenames_Lesion_Mask[0]) + desired_part1 + "_" + desired_part2 + ".nii"
        Fold_Lesion_Mask_pathes.append(desired_part)

    return Fold_Lesion_Mask_pathes


# This function performs both Re-slicing and cropping (written by Moslem)
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
    # Create the ResampleImageFilter
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(list(new_spacing))
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
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


# This function performs slicing on each MRI volume in z axis (written by Moslem)
def slice_data(image, image_name, fold_number, new_spacing, crop_size):
    # image: input simpleitk image list
    # image_name: could be one of "T2w", "Adc", "Hbv", "Lesion" or "Prostate"
    # fold_number: could be one of 0, 1, 2, 3, 4
    # new_spacing: desired(output) spacing
    # crop_size: desired(output) image size

    if image_name == "T2w":
        var_name = "image_T2w_list_Fold_" + str(fold_number)
        dict_vars = {}
        dict_vars[var_name] = []
        for i in range(len(image)):
            resample_image_T2w = volume_resample_crop(image[i], new_spacing, crop_size, image_name)
            for z in range(resample_image_T2w.GetSize()[2]):
                z_slice = sitk.Extract(resample_image_T2w, [resample_image_T2w.GetSize()[0],
                                                            resample_image_T2w.GetSize()[1], 0],
                                       [0, 0, z])
                dict_vars[var_name].append(z_slice)
    elif image_name == "Adc":
        var_name = "image_Adc_list_Fold_" + str(fold_number)
        dict_vars = {}
        dict_vars[var_name] = []
        for i in range(len(image)):
            resample_image_Adc = volume_resample_crop(image[i], new_spacing, crop_size, image_name)
            for z in range(resample_image_Adc.GetSize()[2]):
                z_slice = sitk.Extract(resample_image_Adc, [resample_image_Adc.GetSize()[0],
                                                            resample_image_Adc.GetSize()[1], 0],
                                       [0, 0, z])
                dict_vars[var_name].append(z_slice)
    elif image_name == "Hbv":
        var_name = "image_Hbv_list_Fold_" + str(fold_number)
        dict_vars = {}
        dict_vars[var_name] = []
        for i in range(len(image)):
            resample_image_Hbv = volume_resample_crop(image[i], new_spacing, crop_size, image_name)
            for z in range(resample_image_Hbv.GetSize()[2]):
                z_slice = sitk.Extract(resample_image_Hbv, [resample_image_Hbv.GetSize()[0],
                                                            resample_image_Hbv.GetSize()[1], 0],
                                       [0, 0, z])
                dict_vars[var_name].append(z_slice)
    elif image_name == "Lesion":
        var_name = "image_Lesion_list_Fold_" + str(fold_number)
        dict_vars = {}
        dict_vars[var_name] = []
        for i in range(len(image)):
            resample_image_Lesion = volume_resample_crop(image[i], new_spacing, crop_size, image_name)
            for z in range(resample_image_Lesion.GetSize()[2]):
                z_slice = sitk.Extract(resample_image_Lesion, [resample_image_Lesion.GetSize()[0],
                                                               resample_image_Lesion.GetSize()[1], 0],
                                       [0, 0, z])
                dict_vars[var_name].append(z_slice)
    elif image_name == "Prostate":
        var_name = "image_Prostate_list_Fold_" + str(fold_number)
        dict_vars = {}
        dict_vars[var_name] = []
        for i in range(len(image)):
            resample_image_Prostate = volume_resample_crop(image[i], new_spacing, crop_size, image_name)
            for z in range(resample_image_Prostate.GetSize()[2]):
                z_slice = sitk.Extract(resample_image_Prostate, [resample_image_Prostate.GetSize()[0],
                                                                 resample_image_Prostate.GetSize()[1], 0],
                                       [0, 0, z])
                dict_vars[var_name].append(z_slice)

    return dict_vars[var_name]


# This function calculate mean and std of training dataset
def compute_mean_and_std(train_ds):
    for i in tqdm(range(len(train_ds))):
        image = train_ds[i]
        pixel_array = np.array(image)
        if i == 0:
            mean = pixel_array.mean(axis=(0, 1))
            std = pixel_array.std(axis=(0, 1))
        else:
            mean += pixel_array.mean(axis=(0, 1))
            std += pixel_array.std(axis=(0, 1))

    mean /= len(train_ds)
    std /= len(train_ds)

    return mean, std


# This function performs augmentation and normalization for training set (written by Moslem)
def augment_data(img_list, data_mean, data_std, data_type):
    # img_list: input numpy image list
    # data_mean: mean of data to be normalized
    # data_std: std of data to be normalized
    # data_type: could be one of "data" or "target"

    if data_type == "data":
        transform_tr = T.Compose([
            # T.RandomHorizontalFlip(1.0),
            T.Normalize(data_mean, data_std)
        ])
    elif data_type == "target":
        transform_tr = T.Compose([
            # T.RandomHorizontalFlip(1.0)
        ])
    transformed_imgs = []
    for img in img_list:
        float_tensor = T.ToTensor()(Image.fromarray(img)).float()
        img_tensor = transform_tr(float_tensor)
        transformed_imgs.append(img_tensor)
    return transformed_imgs


# This function performs augmentation and normalization for validation and test sets (written by Moslem)
def augment_data_test(img_list, data_mean, data_std, data_type):
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


# This function simply create folders based on given path
def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


# This function perform training and validation (written by Moslem)
def training_loop(args, model, train_loader, val_loader, device):
    train_loss = []
    validation_loss = []
    log_direction = os.path.join(args.log_dir, args.data)
    mkdir_if_missing(log_direction)
    optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    global_step = 0

    # Begin training
    best_train_score = 0
    best_val_score = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        # tqdm visualization of progress bar
        with tqdm(total=len(train_loader.dataset), desc=f'Epoch {epoch + 1}/{args.epochs}', unit='img') as pbar:
            for batch in train_loader:
                # extract the input images and masks in a batch
                images = batch[0]
                true_masks = batch[1]

                # check if the channel info has been set correctly
                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                # calculate loss based on the prediction
                with torch.cuda.amp.autocast(enabled=args.amp):
                    masks_pred = model(images)
                    loss = dice_loss(F.softmax(masks_pred, dim=1).float(),
                                     F.one_hot(torch.squeeze(true_masks), model.n_classes).permute(0, 3, 1, 2).float(),
                                     multiclass=True)

                # optimizer set-up
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                # loss accumulation and step calculation
                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})

        # calculate Dice score at end of each epoch
        val_score = evaluate(model, val_loader, device, args.amp)
        train_score = evaluate(model, train_loader, device, args.amp)
        scheduler.step(val_score)
        validation_loss.append((1.0 - val_score.cpu().numpy()))
        train_loss.append((1.0 - train_score.cpu().numpy()))
        # general check point save
        if epoch % args.save_step == 0 or epoch == args.epochs - 1:
            torch.save(model.state_dict(), os.path.join(log_direction, 'checkpoint_epoch_{}.pth'.format(epoch + 1)))

        # update and save the best checkpoint
        if train_score >= best_train_score and val_score >= best_val_score:
            best_train_score = train_score
            best_val_score = val_score
            torch.save(model.state_dict(), os.path.join(log_direction, 'best.pth'))

    # evaluate Dice score for training/validation/testing datasets
    print("start of evaluation loop")
    val_score = evaluate(model, val_loader, device, args.amp)
    train_score = evaluate(model, train_loader, device, args.amp)
    test_score = evaluate(model, test_loader, device, args.amp)

    # report the evaluated Dice score for training/validation/testing set at the end of training
    print('Training Dice score: {}'.format(train_score))
    print('Validation Dice score: {}'.format(val_score))
    print('Test Dice score: {}'.format(test_score))
    return train_loss, validation_loss


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Assignment 3')
    # hyper-parameters
    parser.add_argument('-learning_rate', type=float, default=1e-5, help="learning rate")
    parser.add_argument('-BatchSize', default=8, type=int, metavar='N', help='mini-batch size Default: 8')
    parser.add_argument('-data', default='T2w', help='Data_type: choose among "Adc", "T2w", and "Hbv"')
    parser.add_argument('-loss', default='dice_loss', help='loss for training network')
    parser.add_argument('-epochs', default=50, type=int, metavar='N', help='epochs for training process')
    parser.add_argument('-save_step', default=10, type=int, metavar='N', help='number of epochs to save model')
    parser.add_argument('-log_dir', default='checkpoints', help='where the trained models save')
    parser.add_argument('-data_path', default='D:/Courses/CISC_881_Medical_Imaging/PICAI_dataset/', help='path to data')
    parser.add_argument('--Scheduler', type=bool, default=False, help='Change learning rate based on a scheduler')
    parser.add_argument('--amp', type=bool, default=False, help='Change learning rate based on a scheduler')
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--weight-decay', type=float, default=1e-6)
    parser.add_argument('--min_delta', type=float, default=0.07, help='where the trained models save')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    args = parser.parse_args()

    # set the size and spacing for all data
    output_spacing = [0.5, 0.5, 3.0]
    output_size = [300, 300, 16]

    if args.data == "T2w":

        # Set the file path of T2w images and prostate masks for each fold (change these based on your data directory)
        Filenames_Images_T2w_Fold_0 = glob.glob(os.path.join(args.data_path, "picai_public_images_fold0/*/*t2w.mha"))
        Filenames_Images_T2w_Fold_1 = glob.glob(os.path.join(args.data_path, "picai_public_images_fold1/*/*t2w.mha"))
        Filenames_Images_T2w_Fold_2 = glob.glob(os.path.join(args.data_path, "picai_public_images_fold2/*/*t2w.mha"))
        Filenames_Images_T2w_Fold_3 = glob.glob(os.path.join(args.data_path, "picai_public_images_fold3/*/*t2w.mha"))
        Filenames_Images_T2w_Fold_4 = glob.glob(os.path.join(args.data_path, "picai_public_images_fold4/*/*t2w.mha"))
        Filenames_Prostate_Mask = glob.glob(os.path.join(args.data_path,
                                                         "picai_labels-main/anatomical_delineations/whole_gland/AI"
                                                         "/Bosma22b/"))

        Filenames_Prostate_Mask_Fold_0 = read_folds(Filenames_Images_T2w_Fold_0, Filenames_Prostate_Mask)
        Filenames_Prostate_Mask_Fold_1 = read_folds(Filenames_Images_T2w_Fold_1, Filenames_Prostate_Mask)
        Filenames_Prostate_Mask_Fold_2 = read_folds(Filenames_Images_T2w_Fold_2, Filenames_Prostate_Mask)
        Filenames_Prostate_Mask_Fold_3 = read_folds(Filenames_Images_T2w_Fold_3, Filenames_Prostate_Mask)
        Filenames_Prostate_Mask_Fold_4 = read_folds(Filenames_Images_T2w_Fold_4, Filenames_Prostate_Mask)

        # Reading T2w images with sitk
        Images_T2w_Fold_0 = [sitk.ReadImage(filename) for filename in Filenames_Images_T2w_Fold_0]
        Images_T2w_Fold_1 = [sitk.ReadImage(filename) for filename in Filenames_Images_T2w_Fold_1]
        Images_T2w_Fold_2 = [sitk.ReadImage(filename) for filename in Filenames_Images_T2w_Fold_2]
        Images_T2w_Fold_3 = [sitk.ReadImage(filename) for filename in Filenames_Images_T2w_Fold_3]
        Images_T2w_Fold_4 = [sitk.ReadImage(filename) for filename in Filenames_Images_T2w_Fold_4]

        # Reading prostate masks with sitk
        Prostate_Mask_Fold_0 = [sitk.ReadImage(filename) for filename in Filenames_Prostate_Mask_Fold_0]
        Prostate_Mask_Fold_1 = [sitk.ReadImage(filename) for filename in Filenames_Prostate_Mask_Fold_1]
        Prostate_Mask_Fold_2 = [sitk.ReadImage(filename) for filename in Filenames_Prostate_Mask_Fold_2]
        Prostate_Mask_Fold_3 = [sitk.ReadImage(filename) for filename in Filenames_Prostate_Mask_Fold_3]
        Prostate_Mask_Fold_4 = [sitk.ReadImage(filename) for filename in Filenames_Prostate_Mask_Fold_4]

        # apply the desirable size and spacing with slicing the z dimension
        image_T2w_Fold_0_list = slice_data(Images_T2w_Fold_0, "T2w", 0, output_spacing, output_size)
        del Images_T2w_Fold_0
        image_T2w_Fold_1_list = slice_data(Images_T2w_Fold_1, "T2w", 1, output_spacing, output_size)
        del Images_T2w_Fold_1
        image_T2w_Fold_2_list = slice_data(Images_T2w_Fold_2, "T2w", 2, output_spacing, output_size)
        del Images_T2w_Fold_2
        image_T2w_Fold_3_list = slice_data(Images_T2w_Fold_3, "T2w", 3, output_spacing, output_size)
        del Images_T2w_Fold_3
        image_T2w_Fold_4_list = slice_data(Images_T2w_Fold_4, "T2w", 4, output_spacing, output_size)
        del Images_T2w_Fold_4
        image_Prostate_Fold_0_list = slice_data(Prostate_Mask_Fold_0, "Prostate", 0, output_spacing, output_size)
        del Prostate_Mask_Fold_0
        image_Prostate_Fold_1_list = slice_data(Prostate_Mask_Fold_1, "Prostate", 1, output_spacing, output_size)
        del Prostate_Mask_Fold_1
        image_Prostate_Fold_2_list = slice_data(Prostate_Mask_Fold_2, "Prostate", 2, output_spacing, output_size)
        del Prostate_Mask_Fold_2
        image_Prostate_Fold_3_list = slice_data(Prostate_Mask_Fold_3, "Prostate", 3, output_spacing, output_size)
        del Prostate_Mask_Fold_3
        image_Prostate_Fold_4_list = slice_data(Prostate_Mask_Fold_4, "Prostate", 4, output_spacing, output_size)
        del Prostate_Mask_Fold_4

        # creating train, validation and test data and convert simpleitk to numpy
        training_Tw2 = image_T2w_Fold_1_list + image_T2w_Fold_2_list + image_T2w_Fold_4_list
        val_Tw2 = image_T2w_Fold_3_list
        test_Tw2 = image_T2w_Fold_0_list
        training_Tw2_numpy = []
        val_Tw2_numpy = []
        test_Tw2_numpy = []
        for sitk_image in training_Tw2:
            np_image = sitk.GetArrayFromImage(sitk_image)
            training_Tw2_numpy.append(np_image)
        del training_Tw2
        for sitk_image in val_Tw2:
            np_image = sitk.GetArrayFromImage(sitk_image)
            val_Tw2_numpy.append(np_image)
        del val_Tw2
        for sitk_image in test_Tw2:
            np_image = sitk.GetArrayFromImage(sitk_image)
            test_Tw2_numpy.append(np_image)
        del test_Tw2
        training_target_Tw2 = image_Prostate_Fold_1_list + image_Prostate_Fold_2_list + image_Prostate_Fold_4_list
        val_target_Tw2 = image_Prostate_Fold_3_list
        test_target_Tw2 = image_Prostate_Fold_0_list
        training_target_Tw2_numpy = []
        val_target_Tw2_numpy = []
        test_target_Tw2_numpy = []
        for sitk_image in training_target_Tw2:
            np_image = sitk.GetArrayFromImage(sitk_image)
            training_target_Tw2_numpy.append(np_image)
        del training_target_Tw2
        for sitk_image in val_target_Tw2:
            np_image = sitk.GetArrayFromImage(sitk_image)
            val_target_Tw2_numpy.append(np_image)
        del val_target_Tw2
        for sitk_image in test_target_Tw2:
            np_image = sitk.GetArrayFromImage(sitk_image)
            test_target_Tw2_numpy.append(np_image)
        del test_target_Tw2

        # # calculate the mean and std of T2w training data
        # mean_T2w, std_T2w = compute_mean_and_std(training_Tw2_numpy) # Uncomment for calculating mean and std
        mean_T2w = 209.71
        std_T2w = 134.86

        # Apply the transforms (augmentation and normalization)
        transformed_training_imgs = augment_data(training_Tw2_numpy, mean_T2w, std_T2w, "data")
        del training_Tw2_numpy
        transformed_val_imgs = augment_data_test(val_Tw2_numpy, mean_T2w, std_T2w, "data")
        del val_Tw2_numpy
        transformed_test_imgs = augment_data_test(test_Tw2_numpy, mean_T2w, std_T2w, "data")
        del test_Tw2_numpy
        transformed_training_targets = augment_data(training_target_Tw2_numpy, mean_T2w, std_T2w, "target")
        del training_target_Tw2_numpy
        transformed_val_targets = augment_data_test(val_target_Tw2_numpy, mean_T2w, std_T2w, "target")
        del val_target_Tw2_numpy
        transformed_test_targets = augment_data_test(test_target_Tw2_numpy, mean_T2w, std_T2w, "target")
        del test_target_Tw2_numpy

        # prepare data for the Unet
        train_dataset_Tw2 = TensorDataset(torch.stack(transformed_training_imgs),
                                          torch.stack(transformed_training_targets))
        val_dataset_Tw2 = TensorDataset(torch.stack(transformed_val_imgs), torch.stack(transformed_val_targets))
        test_dataset_Tw2 = TensorDataset(torch.stack(transformed_test_imgs), torch.stack(transformed_test_targets))
        train_loader = torch.utils.data.DataLoader(train_dataset_Tw2, batch_size=args.BatchSize, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset_Tw2, batch_size=args.BatchSize)
        test_loader = torch.utils.data.DataLoader(test_dataset_Tw2, batch_size=args.BatchSize)

    if args.data == 'Adc':

        # Set the file path of Adc images and prostate masks for each fold (change these based on your data directory)
        Filenames_Images_Adc_Fold_0 = glob.glob(os.path.join(args.data_path, "picai_public_images_fold0/*/*adc.mha"))
        Filenames_Images_Adc_Fold_1 = glob.glob(os.path.join(args.data_path, "picai_public_images_fold1/*/*adc.mha"))
        Filenames_Images_Adc_Fold_2 = glob.glob(os.path.join(args.data_path, "picai_public_images_fold2/*/*adc.mha"))
        Filenames_Images_Adc_Fold_3 = glob.glob(os.path.join(args.data_path, "picai_public_images_fold3/*/*adc.mha"))
        Filenames_Images_Adc_Fold_4 = glob.glob(os.path.join(args.data_path, "picai_public_images_fold4/*/*adc.mha"))
        Filenames_Lesion_Mask = glob.glob(os.path.join(args.data_path,
                                                       "picai_labels-main/csPCa_lesion_delineations/AI/Bosma22a/"))
        Filenames_Lesion_Mask_Fold_0 = read_folds(Filenames_Images_Adc_Fold_0, Filenames_Lesion_Mask)
        Filenames_Lesion_Mask_Fold_1 = read_folds(Filenames_Images_Adc_Fold_1, Filenames_Lesion_Mask)
        Filenames_Lesion_Mask_Fold_2 = read_folds(Filenames_Images_Adc_Fold_2, Filenames_Lesion_Mask)
        Filenames_Lesion_Mask_Fold_3 = read_folds(Filenames_Images_Adc_Fold_3, Filenames_Lesion_Mask)
        Filenames_Lesion_Mask_Fold_4 = read_folds(Filenames_Images_Adc_Fold_4, Filenames_Lesion_Mask)

        # Reading Adc images with sitk
        Images_Adc_Fold_0 = [sitk.ReadImage(filename) for filename in Filenames_Images_Adc_Fold_0]
        Images_Adc_Fold_1 = [sitk.ReadImage(filename) for filename in Filenames_Images_Adc_Fold_1]
        Images_Adc_Fold_2 = [sitk.ReadImage(filename) for filename in Filenames_Images_Adc_Fold_2]
        Images_Adc_Fold_3 = [sitk.ReadImage(filename) for filename in Filenames_Images_Adc_Fold_3]
        Images_Adc_Fold_4 = [sitk.ReadImage(filename) for filename in Filenames_Images_Adc_Fold_4]

        # Reading lesion masks with sitk
        Lesion_Mask_Fold_0 = [sitk.ReadImage(filename) for filename in Filenames_Lesion_Mask_Fold_0]
        Lesion_Mask_Fold_1 = [sitk.ReadImage(filename) for filename in Filenames_Lesion_Mask_Fold_1]
        Lesion_Mask_Fold_2 = [sitk.ReadImage(filename) for filename in Filenames_Lesion_Mask_Fold_2]
        Lesion_Mask_Fold_3 = [sitk.ReadImage(filename) for filename in Filenames_Lesion_Mask_Fold_3]
        Lesion_Mask_Fold_4 = [sitk.ReadImage(filename) for filename in Filenames_Lesion_Mask_Fold_4]

        # apply the desirable size and spacing with slicing the z dimension
        image_Adc_Fold_0_list = slice_data(Images_Adc_Fold_0, "Adc", 0, output_spacing, output_size)
        del Images_Adc_Fold_0
        image_Adc_Fold_1_list = slice_data(Images_Adc_Fold_1, "Adc", 1, output_spacing, output_size)
        del Images_Adc_Fold_1
        image_Adc_Fold_2_list = slice_data(Images_Adc_Fold_2, "Adc", 2, output_spacing, output_size)
        del Images_Adc_Fold_2
        image_Adc_Fold_3_list = slice_data(Images_Adc_Fold_3, "Adc", 3, output_spacing, output_size)
        del Images_Adc_Fold_3
        image_Adc_Fold_4_list = slice_data(Images_Adc_Fold_4, "Adc", 4, output_spacing, output_size)
        del Images_Adc_Fold_4
        image_Lesion_Fold_0_list = slice_data(Lesion_Mask_Fold_0, "Lesion", 0, output_spacing, output_size)
        del Lesion_Mask_Fold_0
        image_Lesion_Fold_1_list = slice_data(Lesion_Mask_Fold_1, "Lesion", 1, output_spacing, output_size)
        del Lesion_Mask_Fold_1
        image_Lesion_Fold_2_list = slice_data(Lesion_Mask_Fold_2, "Lesion", 2, output_spacing, output_size)
        del Lesion_Mask_Fold_2
        image_Lesion_Fold_3_list = slice_data(Lesion_Mask_Fold_3, "Lesion", 3, output_spacing, output_size)
        del Lesion_Mask_Fold_3
        image_Lesion_Fold_4_list = slice_data(Lesion_Mask_Fold_4, "Lesion", 4, output_spacing, output_size)
        del Lesion_Mask_Fold_4

        # creating train, validation and test data and convert simpleitk to numpy
        training_Adc = image_Adc_Fold_1_list + image_Adc_Fold_2_list + image_Adc_Fold_4_list
        val_Adc = image_Adc_Fold_3_list
        test_Adc = image_Adc_Fold_0_list
        training_Adc_numpy = []
        val_Adc_numpy = []
        test_Adc_numpy = []
        for sitk_image in training_Adc:
            np_image = sitk.GetArrayFromImage(sitk_image)
            training_Adc_numpy.append(np_image)
        del training_Adc
        for sitk_image in val_Adc:
            np_image = sitk.GetArrayFromImage(sitk_image)
            val_Adc_numpy.append(np_image)
        del val_Adc
        for sitk_image in test_Adc:
            np_image = sitk.GetArrayFromImage(sitk_image)
            test_Adc_numpy.append(np_image)
        del test_Adc
        training_target_Hbv_Adc = image_Lesion_Fold_1_list + image_Lesion_Fold_2_list + image_Lesion_Fold_4_list
        val_target_Hbv_Adc = image_Lesion_Fold_3_list
        test_target_Hbv_Adc = image_Lesion_Fold_0_list
        training_target_Hbv_Adc_numpy = []
        val_target_Hbv_Adc_numpy = []
        test_target_Hbv_Adc_numpy = []
        for sitk_image in training_target_Hbv_Adc:
            np_image = sitk.GetArrayFromImage(sitk_image)
            training_target_Hbv_Adc_numpy.append(np_image)
        del training_target_Hbv_Adc
        for sitk_image in val_target_Hbv_Adc:
            np_image = sitk.GetArrayFromImage(sitk_image)
            val_target_Hbv_Adc_numpy.append(np_image)
        del val_target_Hbv_Adc
        for sitk_image in test_target_Hbv_Adc:
            np_image = sitk.GetArrayFromImage(sitk_image)
            test_target_Hbv_Adc_numpy.append(np_image)
        del test_target_Hbv_Adc

        # # calculate the mean and std of Adc training data
        # mean_Adc, std_Adc = compute_mean_and_std(training_Adc_numpy) # Uncomment for calculating mean and std
        mean_Adc = 777.54
        std_Adc = 701.82

        # Apply the transforms (augmentation and normalization)
        transformed_training_imgs = augment_data(training_Adc_numpy, mean_Adc, std_Adc, "data")
        del training_Adc_numpy
        transformed_val_imgs = augment_data_test(val_Adc_numpy, mean_Adc, std_Adc, "data")
        del val_Adc_numpy
        transformed_test_imgs = augment_data_test(test_Adc_numpy, mean_Adc, std_Adc, "data")
        del test_Adc_numpy
        transformed_training_targets = augment_data(training_target_Hbv_Adc_numpy, mean_Adc, std_Adc, "target")
        del training_target_Hbv_Adc_numpy
        transformed_val_targets = augment_data_test(val_target_Hbv_Adc_numpy, mean_Adc, std_Adc, "target")
        del val_target_Hbv_Adc_numpy
        transformed_test_targets = augment_data_test(test_target_Hbv_Adc_numpy, mean_Adc, std_Adc, "target")
        del test_target_Hbv_Adc_numpy

        # prepare data for the Unet
        train_dataset_Adc = TensorDataset(torch.stack(transformed_training_imgs),
                                          torch.stack(transformed_training_targets))
        val_dataset_Adc = TensorDataset(torch.stack(transformed_val_imgs), torch.stack(transformed_val_targets))
        test_dataset_Adc = TensorDataset(torch.stack(transformed_test_imgs), torch.stack(transformed_test_targets))
        train_loader = torch.utils.data.DataLoader(train_dataset_Adc, batch_size=args.BatchSize, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset_Adc, batch_size=args.BatchSize, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset_Adc, batch_size=args.BatchSize, shuffle=False)

    if args.data == "Hbv":

        # Set the file path of Hbv images and prostate masks for each fold (change these based on your data directory)
        Filenames_Images_Hbv_Fold_0 = glob.glob(os.path.join(args.data_path, "picai_public_images_fold0/*/*hbv.mha"))
        Filenames_Images_Hbv_Fold_1 = glob.glob(os.path.join(args.data_path, "picai_public_images_fold1/*/*hbv.mha"))
        Filenames_Images_Hbv_Fold_2 = glob.glob(os.path.join(args.data_path, "picai_public_images_fold2/*/*hbv.mha"))
        Filenames_Images_Hbv_Fold_3 = glob.glob(os.path.join(args.data_path, "picai_public_images_fold3/*/*hbv.mha"))
        Filenames_Images_Hbv_Fold_4 = glob.glob(os.path.join(args.data_path, "picai_public_images_fold4/*/*hbv.mha"))
        Filenames_Lesion_Mask = glob.glob(os.path.join(args.data_path, "picai_labels-main/csPCa_lesion_delineations/AI"
                                                                       "/Bosma22a/"))
        Filenames_Lesion_Mask_Fold_0 = read_folds(Filenames_Images_Hbv_Fold_0, Filenames_Lesion_Mask)
        Filenames_Lesion_Mask_Fold_1 = read_folds(Filenames_Images_Hbv_Fold_1, Filenames_Lesion_Mask)
        Filenames_Lesion_Mask_Fold_2 = read_folds(Filenames_Images_Hbv_Fold_2, Filenames_Lesion_Mask)
        Filenames_Lesion_Mask_Fold_3 = read_folds(Filenames_Images_Hbv_Fold_3, Filenames_Lesion_Mask)
        Filenames_Lesion_Mask_Fold_4 = read_folds(Filenames_Images_Hbv_Fold_4, Filenames_Lesion_Mask)

        # Reading Hbv images with sitk
        Images_Hbv_Fold_0 = [sitk.ReadImage(filename) for filename in Filenames_Images_Hbv_Fold_0]
        Images_Hbv_Fold_1 = [sitk.ReadImage(filename) for filename in Filenames_Images_Hbv_Fold_1]
        Images_Hbv_Fold_2 = [sitk.ReadImage(filename) for filename in Filenames_Images_Hbv_Fold_2]
        Images_Hbv_Fold_3 = [sitk.ReadImage(filename) for filename in Filenames_Images_Hbv_Fold_3]
        Images_Hbv_Fold_4 = [sitk.ReadImage(filename) for filename in Filenames_Images_Hbv_Fold_4]

        # Reading lesion masks with sitk
        Lesion_Mask_Fold_0 = [sitk.ReadImage(filename) for filename in Filenames_Lesion_Mask_Fold_0]
        Lesion_Mask_Fold_1 = [sitk.ReadImage(filename) for filename in Filenames_Lesion_Mask_Fold_1]
        Lesion_Mask_Fold_2 = [sitk.ReadImage(filename) for filename in Filenames_Lesion_Mask_Fold_2]
        Lesion_Mask_Fold_3 = [sitk.ReadImage(filename) for filename in Filenames_Lesion_Mask_Fold_3]
        Lesion_Mask_Fold_4 = [sitk.ReadImage(filename) for filename in Filenames_Lesion_Mask_Fold_4]

        # apply the desirable size and spacing with slicing the z dimension
        image_Hbv_Fold_0_list = slice_data(Images_Hbv_Fold_0, "Hbv", 0, output_spacing, output_size)
        del Images_Hbv_Fold_0
        image_Hbv_Fold_1_list = slice_data(Images_Hbv_Fold_1, "Hbv", 1, output_spacing, output_size)
        del Images_Hbv_Fold_1
        image_Hbv_Fold_2_list = slice_data(Images_Hbv_Fold_2, "Hbv", 2, output_spacing, output_size)
        del Images_Hbv_Fold_2
        image_Hbv_Fold_3_list = slice_data(Images_Hbv_Fold_3, "Hbv", 3, output_spacing, output_size)
        del Images_Hbv_Fold_3
        image_Hbv_Fold_4_list = slice_data(Images_Hbv_Fold_4, "Hbv", 4, output_spacing, output_size)
        del Images_Hbv_Fold_4
        image_Lesion_Fold_0_list = slice_data(Lesion_Mask_Fold_0, "Lesion", 0, output_spacing, output_size)
        del Lesion_Mask_Fold_0
        image_Lesion_Fold_1_list = slice_data(Lesion_Mask_Fold_1, "Lesion", 1, output_spacing, output_size)
        del Lesion_Mask_Fold_1
        image_Lesion_Fold_2_list = slice_data(Lesion_Mask_Fold_2, "Lesion", 2, output_spacing, output_size)
        del Lesion_Mask_Fold_2
        image_Lesion_Fold_3_list = slice_data(Lesion_Mask_Fold_3, "Lesion", 3, output_spacing, output_size)
        del Lesion_Mask_Fold_3
        image_Lesion_Fold_4_list = slice_data(Lesion_Mask_Fold_4, "Lesion", 4, output_spacing, output_size)
        del Lesion_Mask_Fold_4

        # creating train, validation and test data and convert simpleitk to numpy
        training_Hbv = image_Hbv_Fold_1_list + image_Hbv_Fold_2_list + image_Hbv_Fold_4_list
        val_Hbv = image_Hbv_Fold_3_list
        test_Hbv = image_Hbv_Fold_0_list
        training_Hbv_numpy = []
        val_Hbv_numpy = []
        test_Hbv_numpy = []
        for sitk_image in training_Hbv:
            np_image = sitk.GetArrayFromImage(sitk_image)
            training_Hbv_numpy.append(np_image)
        del training_Hbv
        for sitk_image in val_Hbv:
            np_image = sitk.GetArrayFromImage(sitk_image)
            val_Hbv_numpy.append(np_image)
        del val_Hbv
        for sitk_image in test_Hbv:
            np_image = sitk.GetArrayFromImage(sitk_image)
            test_Hbv_numpy.append(np_image)
        del test_Hbv
        training_target_Hbv_Adc = image_Lesion_Fold_1_list + image_Lesion_Fold_2_list + image_Lesion_Fold_4_list
        val_target_Hbv_Adc = image_Lesion_Fold_3_list
        test_target_Hbv_Adc = image_Lesion_Fold_0_list
        training_target_Hbv_Adc_numpy = []
        val_target_Hbv_Adc_numpy = []
        test_target_Hbv_Adc_numpy = []
        for sitk_image in training_target_Hbv_Adc:
            np_image = sitk.GetArrayFromImage(sitk_image)
            training_target_Hbv_Adc_numpy.append(np_image)
        del training_target_Hbv_Adc
        for sitk_image in val_target_Hbv_Adc:
            np_image = sitk.GetArrayFromImage(sitk_image)
            val_target_Hbv_Adc_numpy.append(np_image)
        del val_target_Hbv_Adc
        for sitk_image in test_target_Hbv_Adc:
            np_image = sitk.GetArrayFromImage(sitk_image)
            test_target_Hbv_Adc_numpy.append(np_image)
        del test_target_Hbv_Adc

        # # calculate the mean and std of Hbv training data
        # mean_Hbv, std_Hbv = compute_mean_and_std(training_Hbv_numpy) # Uncomment for calculating mean and std
        mean_Hbv = 11.81
        std_Hbv = 7.65

        # Apply the transforms (augmentation and normalization)
        transformed_training_imgs = augment_data(training_Hbv_numpy, mean_Hbv, std_Hbv, "data")
        del training_Hbv_numpy
        transformed_val_imgs = augment_data_test(val_Hbv_numpy, mean_Hbv, std_Hbv, "data")
        del val_Hbv_numpy
        transformed_test_imgs = augment_data_test(test_Hbv_numpy, mean_Hbv, std_Hbv, "data")
        del test_Hbv_numpy
        transformed_training_targets = augment_data(training_target_Hbv_Adc_numpy, mean_Hbv, std_Hbv, "target")
        del training_target_Hbv_Adc_numpy
        transformed_val_targets = augment_data_test(val_target_Hbv_Adc_numpy, mean_Hbv, std_Hbv, "target")
        del val_target_Hbv_Adc_numpy
        transformed_test_targets = augment_data_test(test_target_Hbv_Adc_numpy, mean_Hbv, std_Hbv, "target")
        del test_target_Hbv_Adc_numpy

        # prepare data for the Unet
        train_dataset_Hbv = TensorDataset(torch.stack(transformed_training_imgs),
                                          torch.stack(transformed_training_targets))
        val_dataset_Hbv = TensorDataset(torch.stack(transformed_val_imgs), torch.stack(transformed_val_targets))
        test_dataset_Hbv = TensorDataset(torch.stack(transformed_test_imgs), torch.stack(transformed_test_targets))
        train_loader = torch.utils.data.DataLoader(train_dataset_Hbv, batch_size=args.BatchSize, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset_Hbv, batch_size=args.BatchSize)
        test_loader = torch.utils.data.DataLoader(test_dataset_Hbv, batch_size=args.BatchSize)

    # Create the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=1, n_classes=2, bilinear=args.bilinear)
    # print the network summary
    print(model)
    model.to(device)
    print("start of training loop")
    # Perform train and validation loop
    train_loss_list, val_loss_list = training_loop(args, model, train_loader, val_loader, device)

    # Plot Losses after training
    plt.plot(train_loss_list, label='Training loss')
    plt.plot(val_loss_list, label='Validation loss')
    plt.legend()
    plt.title("Dic_Loss")
    plt.grid()
    plt.ylim(0, 1)
    plt.show()
