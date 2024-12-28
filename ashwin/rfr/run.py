import argparse
import os
from model import RFRNetModel
from dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np

from torch.utils.data import TensorDataset, DataLoader, ConcatDataset

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F





def apply_masks_to_images(images, masks):
    # Ensure the images and masks arrays have the same shape
    assert images.shape[0] == masks.shape[0]
    
    # Expand the dimensions of the masks to match the shape of the images
    expanded_masks = np.expand_dims(masks, axis=-1)
    
    # Apply the masks to the images
    masked_images = images * expanded_masks
    
    return masked_images



def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--mask_root', type=str)
    parser.add_argument('--model_save_path', type=str, default='checkpoint')
    parser.add_argument('--result_save_path', type=str, default='results')
    parser.add_argument('--target_size', type=int, default=256)
    parser.add_argument('--mask_mode', type=int, default=1)
    parser.add_argument('--num_iters', type=int, default=450000)
    parser.add_argument('--model_path', type=str, default="checkpoint/100000.pth")
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--n_threads', type=int, default=6)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--gpu_id', type=str, default="0")
    args = parser.parse_args()
        
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    model = RFRNetModel()
    if args.test:
        for epochs in range(20000, 20010, 500):
            model_path = f"checkpoint_size_12/g_f_{epochs}.pth"
            model.initialize_model(model_path, False)
            model.cuda()

            print(args.data_root, 'hi2')

            test_data = np.load(args.data_root)[800:1000]/256
            mask_data = np.load(args.mask_root)[800:1000]
        # test_data = np.repeat(test_data[np.newaxis, ...], 10, axis=0)
        # mask_data = np.repeat(mask_data[np.newaxis, ...], 10, axis=0)


            converted_masks = np.expand_dims(mask_data, axis=-1)
            converted_masks = np.repeat(converted_masks, 3, axis=-1)


            print(test_data.shape, converted_masks.shape)
            size = args.target_size
            masks_tensor = torch.tensor(converted_masks, dtype=torch.float).permute(0,3,1,2)
            images_tensor = torch.tensor(test_data, dtype=torch.float).permute(0,3,1,2)
            my_data = TensorDataset(images_tensor, masks_tensor)
            dataloader = DataLoader(my_data, batch_size = args.batch_size, shuffle = True, num_workers = args.n_threads)

        # dataloader = DataLoader(Dataset(args.data_root, args.mask_root, args.mask_mode, args.target_size, mask_reverse = True, training=False))
            model.test(dataloader, args.result_save_path)

    else:
        model.initialize_model(args.model_path, True)
        model.cuda()
        print(args.data_root, 'hi')
        train_data = np.load(args.data_root)[0:800]/256
        mask_data = np.load(args.mask_root)[:800]
        # plt.imshow(train_data)
        # plt.savefig('testing_img.png')
        # plt.imshow(mask_data)
        # plt.savefig('testing_mask.png')

        # train_data = np.repeat(train_data[np.newaxis, ...], 200, axis=0)
        # mask_data = np.repeat(mask_data[np.newaxis, ...], 200, axis=0)

        converted_masks = np.expand_dims(mask_data, axis=-1)
        
        # Repeat the mask along the new axis to create multiple channels
        converted_masks = np.repeat(converted_masks, 3, axis=-1)


        print(train_data.shape, converted_masks.shape)
        # masked_images = apply_masks_to_images(train_data, mask_data)

        size = args.target_size
        print(train_data.shape)
        # plt.imshow(train_data[0,:,:,:])
        # plt.savefig("testing1.png")
        masks_tensor = torch.tensor(converted_masks, dtype=torch.float).permute(0,3,1,2)
        images_tensor = torch.tensor(train_data, dtype=torch.float).permute(0,3,1,2)
        # print(images_tensor[0,:,:,:].shape)
        # numpy_image = images_tensor[0,:,:,:].permute(1, 2, 0).cpu().numpy()
        # plt.imshow(numpy_image)
        # plt.savefig("testing2.png")

        # exit(0)
        my_data = TensorDataset(images_tensor, masks_tensor)
        dataloader = DataLoader(my_data, batch_size = args.batch_size, shuffle = True, num_workers = args.n_threads)

            # dataloader = DataLoader(Dataset(args.data_root, args.mask_root, args.mask_mode, args.target_size, mask_reverse = True, augment = False), batch_size = args.batch_size, shuffle = True, num_workers = args.n_threads)
        
        model.train(dataloader, args.model_save_path, args.finetune, args.num_iters)

if __name__ == '__main__':
    run()