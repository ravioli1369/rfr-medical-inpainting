import torch
import torch.optim as optim
from utils.io import load_ckpt
from utils.io import save_ckpt
from torchvision.utils import make_grid
from torchvision.utils import save_image
from modules.RFRNet import RFRNet, VGG16FeatureExtractor
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim



def calculate_l1(ground_truth, prediction):
    return np.sum(np.abs(ground_truth-prediction))

def calculate_ssim(ground_truth, prediction, mask):
    # Compute SSIM between two images only in the masked region
    # masked_ground_truth = ground_truth * mask
    # score_baseline = structural_similarity(masked_ground_truth, ground_truth,  data_range=np.max(ground_truth) - np.min(ground_truth))
    # ground_truth = ground_truth[np.where(mask==0)]
    # prediction = prediction[np.where(mask==0)]
    # score = structural_similarity(ground_truth, prediction, 
    #                               data_range=np.max(prediction) - np.min(prediction), 
    #                               mask=mask)
    mask = 1-mask
    val, smap = ssim(ground_truth,prediction,data_range=1,gaussian_weights=True,sigma=4, full=True)
    smap_masked = np.multiply(smap,mask)
    # return val
    return np.sum(smap_masked)/np.sum(mask)

    # normalized_score = (score-score_baseline)/(1-score_baseline)
    # print(score)
    return score

def calculate_psnr(ground_truth, prediction, mask):
    # Compute PSNR between two images only in the masked region
    ground_truth = ground_truth[np.where(mask==0)]
    prediction = prediction[np.where(mask==0)]
    mse = np.mean((ground_truth - prediction) ** 2)
    if mse == 0:
        return 100
    max_pixel = np.max(prediction)
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_rmse(ground_truth, prediction, mask):
    # Compute RMSE between two images only in the masked region
    ground_truth = ground_truth[np.where(mask==0)]
    prediction = prediction[np.where(mask==0)]
    rmse = np.sqrt(np.mean((ground_truth - prediction) ** 2))
    return rmse









class RFRNetModel():
    def __init__(self):
        self.G = None
        self.lossNet = None
        self.iter = None
        self.optm_G = None
        self.device = None
        self.real_A = None
        self.real_B = None
        self.fake_B = None
        self.comp_B = None
        self.l1_loss_val = 0.0
    
    def initialize_model(self, path=None, train=True):
        self.G = RFRNet()
        self.lr = 1e-4
        self.optm_G = optim.Adam(self.G.parameters(), lr = self.lr)
        if train:
            self.lossNet = VGG16FeatureExtractor()
        try:
            start_iter = load_ckpt(path, [('generator', self.G)], [('optimizer_G', self.optm_G)])
            if train:
                self.optm_G = optim.Adam(self.G.parameters(), lr = 1e-4)
                print('Model Initialized, iter: ', start_iter)
                self.iter = start_iter
        except:
            print('No trained model, from start')
            self.iter = 0
        
    def cuda(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Model moved to cuda")
            self.G.cuda()
            if self.lossNet is not None:
                self.lossNet.cuda()
        else:
            self.device = torch.device("cpu")
        
    def train(self, train_loader, save_path, finetune = False, iters=10000):
    #    writer = SummaryWriter(log_dir="log_info")
        self.freq_1 = 500
        self.freq_2 = 1000

        self.G.train(finetune = finetune)
        if finetune:
            self.lr = 5e-5
            self.optm_G = optim.Adam(filter(lambda p:p.requires_grad, self.G.parameters()), lr = self.lr)
            self.freq_1=500
            self.freq_2=500
            iters = 25000
        print("Starting training from iteration:{:d}".format(self.iter))
        s_time = time.time()
        self.save_path = save_path
        while self.iter<=iters:
            for items in train_loader:
                gt_images, masks = self.__cuda__(*items)
                masked_images = gt_images * masks
                self.forward(masked_images, masks, gt_images)
                self.update_parameters()
                self.iter += 1
                
                if self.iter % self.freq_1 == 0:
                    e_time = time.time()
                    int_time = e_time - s_time
                    print("Iteration:%d, l1_loss:%.4f, time_taken:%.2f" %(self.iter, self.l1_loss_val/self.freq_1, int_time))
                    s_time = time.time()
                    self.l1_loss_val = 0.0
                
                if self.iter % self.freq_2 == 0:
                    if not os.path.exists('{:s}'.format(save_path)):
                        os.makedirs('{:s}'.format(save_path))
                    if(finetune):
                        save_ckpt('{:s}/g_f_{:d}.pth'.format(save_path, self.iter ), [('generator', self.G)], [('optimizer_G', self.optm_G)], self.iter)
                    
                    else:
                        save_ckpt('{:s}/g_{:d}.pth'.format(save_path, self.iter ), [('generator', self.G)], [('optimizer_G', self.optm_G)], self.iter)
                    
        if not os.path.exists('{:s}'.format(save_path)):
            os.makedirs('{:s}'.format(save_path))
            save_ckpt('{:s}/g_{:s}.pth'.format(save_path, "final"), [('generator', self.G)], [('optimizer_G', self.optm_G)], self.iter)
    
    def test(self, test_loader, result_save_path):
        self.G.eval()
        for para in self.G.parameters():
            para.requires_grad = False
        rmse_arr = []
        ssim_arr = []
        psnr_arr = []
        count = 0
        for items in test_loader:
            gt_images, masks = self.__cuda__(*items)
            masked_images = gt_images * masks
            # print("in test loader")
            # plt.imshow(masks[0,:,:,:].permute(1, 2, 0).cpu().numpy())
            # plt.savefig(f"t{count}.png")
            # masks = torch.cat([masks]*3, dim = 1)
            comp_B = self.forward2(masked_images, masks)
            # fake_B, mask = self.G(masked_images, masks)
            # comp_B = fake_B * (1 - masks) + gt_images * masks
            if not os.path.exists('{:s}/results'.format(result_save_path)):
                os.makedirs('{:s}/results'.format(result_save_path))
            for k in range(comp_B.size(0)):
                count += 1
                og_image = gt_images.permute(2, 3, 1, 0).cpu().detach().numpy()[:,:,:,k]
                my_mask = masks.permute(2, 3, 1, 0).cpu().detach().numpy()[:,:,:,k]
                my_image = comp_B.permute(2, 3, 1, 0).cpu().detach().numpy()[:,:,:,k]
                arr_my_image = np.sum(my_image, axis=2)/3
                arr_gt_image = np.sum(og_image, axis=2)/3
                my_mask = np.sum(my_mask, axis=2)/3
                rmse_val = calculate_rmse(arr_gt_image, arr_my_image, my_mask)
                psnr_val = calculate_psnr(arr_gt_image, arr_my_image, my_mask)
                ssim_val = calculate_ssim(arr_gt_image, arr_my_image, my_mask)
                rmse_arr.append(rmse_val)
                ssim_arr.append(ssim_val)
                psnr_arr.append(psnr_val)
                if not os.path.exists('{:s}'.format(result_save_path)):
                    os.makedirs('{:s}'.format(result_save_path))

                # fig, ax = plt.subplots(1, 3, figsize=(9, 3), dpi=300, sharey=True)
                # ax[0].imshow(arr_my_image, cmap='gray')
                # ax[0].set_title("Inpainted", y=1.02)
                # ax[1].imshow(arr_gt_image, cmap='gray')
                # ax[1].set_title("Ground Truth", y=1.02)
                # img = ax[2].imshow((arr_my_image-arr_gt_image), cmap='gray')
                # ax[2].set_title("Difference", y=1.02)
                # cax = fig.add_axes([0.92, 0.15, 0.01, 0.69])
                # plt.colorbar(img, cax=cax)


                # plt.imshow(arr_my_image, cmap = 'gray')
                # plt.colorbar()
                # plt.savefig(f'{result_save_path}/results/testing_output_test{count}.png')
                # plt.cla()
                # plt.close()
                # plt.imshow(arr_gt_image-arr_my_image, cmap = 'gray')
                # plt.colorbar()
                # plt.savefig(f'{result_save_path}/results/testing_output_diff_test{count}.png')
                # plt.cla()
                # plt.close()

                # grid1 = make_grid(comp_B[k:k+1])
                # file_path = '{:s}/results/img_{:d}.png'.format(result_save_path, count)
                # save_image(grid1, file_path)
                # # print(masked_images.shape, masks.shape)
                # grid2 = make_grid(masked_images[k:k+1] -1 + masks[k:k+1] )
                # file_path = '{:s}/results/masked_img_{:d}.png'.format(result_save_path, count)
                # save_image(grid2, file_path)
                # grid3 = grid1-grid2
                # file_path = '{:s}/results/masked_img_diff_{:d}.png'.format(result_save_path, count)
                # save_image(grid3, file_path)
                # print('saved')
    
        rmse_arr = np.array(rmse_arr)
        psnr_arr = np.array(psnr_arr)
        ssim_arr = np.array(ssim_arr)
        print(np.median(psnr_arr), np.median(rmse_arr), np.median(ssim_arr))


    def forward(self, masked_image, mask, gt_image):
        self.real_A = masked_image
        self.real_B = gt_image
        self.mask = mask
        fake_B, _ = self.G(masked_image, mask)
        self.fake_B = fake_B
        self.comp_B = self.fake_B * (1 - mask) + self.real_B * mask
        # print(self.comp_B.shape)
        # print(com.shape)
        if self.iter % self.freq_1 == 0:
            my_gt = gt_image.permute(2, 3, 1, 0).cpu().detach().numpy()
            com = self.comp_B.permute(2, 3, 1, 0).cpu().detach().numpy()
            arr_mine = com[:,:,:,0]
            arr_og = my_gt[:,:,:,0]
            arr_diff = arr_mine-arr_og
            arr_mine = np.sum(arr_mine, axis=2)/3
            arr_og = np.sum(arr_og, axis=2)/3
            arr_diff = np.sum(arr_diff, axis=2)/3
            if not os.path.exists('{:s}'.format(self.save_path)):
                os.makedirs('{:s}'.format(self.save_path))

            print(len(arr_diff[arr_diff>0.05]))
            # print(com[:,:,:,0]-my_gt[:,:,:,0])
            # print(arr.shape, type(arr[0,0]))
            plt.imshow(arr_diff, cmap = 'gray')
            plt.colorbar()
            plt.savefig(f'{self.save_path}/train_output_diff.png')
            plt.cla()
            plt.close()
            plt.imshow(arr_mine, cmap = 'gray')
            plt.colorbar()
            plt.savefig(f'{self.save_path}/train_output.png')
            plt.cla()
            plt.close()

        # exit(0)

    def forward2(self, masked_image, mask):
        self.real_A = masked_image
        # self.real_B = gt_image
        self.mask = mask
        fake_B, _ = self.G(masked_image, mask)
        self.fake_B = fake_B
        self.comp_B = self.fake_B * (1 - mask) + masked_image
        return self.comp_B



    def update_parameters(self):
        self.update_G()
        self.update_D()
    
    def update_G(self):
        self.optm_G.zero_grad()
        loss_G = self.get_g_loss()
        loss_G.backward()
        self.optm_G.step()
    
    def update_D(self):
        return
    
    def get_g_loss(self):
        real_B = self.real_B
        fake_B = self.fake_B
        comp_B = self.comp_B
        
        real_B_feats = self.lossNet(real_B)
        fake_B_feats = self.lossNet(fake_B)
        comp_B_feats = self.lossNet(comp_B)
        
        tv_loss = self.TV_loss(comp_B * (1 - self.mask))
        style_loss = self.style_loss(real_B_feats, fake_B_feats) + self.style_loss(real_B_feats, comp_B_feats)
        preceptual_loss = self.preceptual_loss(real_B_feats, fake_B_feats) + self.preceptual_loss(real_B_feats, comp_B_feats)
        valid_loss = self.l1_loss(real_B, fake_B, self.mask)
        hole_loss = self.l1_loss(real_B, fake_B, (1 - self.mask))
        
        loss_G = (  tv_loss * 0.1
                  + style_loss * 120
                  + preceptual_loss * 0.05
                  + valid_loss * 1
                  + hole_loss * 6)
        if(self.iter%self.freq_1==0):
            print(tv_loss.detach().cpu()* 0.1, style_loss.detach().cpu()* 120, preceptual_loss.detach().cpu()* 0.05, valid_loss.detach().cpu()* 1, hole_loss.detach().cpu()* 6)
        
        self.l1_loss_val += valid_loss.detach() + hole_loss.detach()
        return loss_G
    
    def l1_loss(self, f1, f2, mask = 1):
        return torch.mean(torch.abs(f1 - f2)*mask)
    
    def style_loss(self, A_feats, B_feats):
        assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]
            _, c, w, h = A_feat.size()
            A_feat = A_feat.view(A_feat.size(0), A_feat.size(1), A_feat.size(2) * A_feat.size(3))
            B_feat = B_feat.view(B_feat.size(0), B_feat.size(1), B_feat.size(2) * B_feat.size(3))
            A_style = torch.matmul(A_feat, A_feat.transpose(2, 1))
            B_style = torch.matmul(B_feat, B_feat.transpose(2, 1))
            loss_value += torch.mean(torch.abs(A_style - B_style)/(c * w * h))
        return loss_value
    
    def TV_loss(self, x):
        h_x = x.size(2)
        w_x = x.size(3)
        h_tv = torch.mean(torch.abs(x[:,:,1:,:]-x[:,:,:h_x-1,:]))
        w_tv = torch.mean(torch.abs(x[:,:,:,1:]-x[:,:,:,:w_x-1]))
        return h_tv + w_tv
    
    def preceptual_loss(self, A_feats, B_feats):
        assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]
            loss_value += torch.mean(torch.abs(A_feat - B_feat))
        return loss_value
            
    def __cuda__(self, *args):
        return (item.to(self.device) for item in args)
            