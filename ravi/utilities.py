import os
import numpy as np
from glob import glob
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity
import matplotlib

params = {
    "text.usetex": True,
    "font.family": "serif",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.top": True,
    "ytick.left": True,
    "ytick.right": True,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.minor.size": 2.5,
    "xtick.major.size": 5,
    "ytick.minor.size": 2.5,
    "ytick.major.size": 5,
    "axes.axisbelow": True,
}
matplotlib.rcParams.update(params)


def create_array_with_zeros(n, hole_ratio, number_of_holes=1):
    # Create an array of ones
    arr = np.ones((n, n))

    # Calculate the size of the square containing zeros
    zero_size = n // hole_ratio

    for _ in range(number_of_holes):
        # Generate random coordinates for the top-left corner of the zero square
        start_row = np.random.randint(0, n - zero_size + 1)
        start_col = np.random.randint(0, n - zero_size + 1)

        # Set the values in the zero square region to zeros
        arr[start_row : start_row + zero_size, start_col : start_col + zero_size] = 0

    return arr


def create_strip_mask(n, thickness, orientation=None, number=1):
    # Create an array of ones
    arr = np.ones((n, n))
    if (number > 1) and (orientation not in ["H", "V"]):
        for _ in range(number // 2):
            start_row = np.random.randint(0, n - thickness + 1)
            arr[start_row : start_row + thickness, :] = 0
            start_col = np.random.randint(0, n - thickness + 1)
            arr[:, start_col : start_col + thickness] = 0
    else:
        for _ in range(number):
            if orientation == "H":
                start_row = np.random.randint(0, n - thickness + 1)
                arr[start_row : start_row + thickness, :] = 0
            elif orientation == "V":
                start_col = np.random.randint(0, n - thickness + 1)
                arr[:, start_col : start_col + thickness] = 0
            else:
                coin_flip = bool(np.random.randint(0, 2))
                if coin_flip:
                    start_row = np.random.randint(0, n - thickness + 1)
                    arr[start_row : start_row + thickness, :] = 0
                else:
                    start_col = np.random.randint(0, n - thickness + 1)
                    arr[:, start_col : start_col + thickness] = 0
    return arr


def calculate_ssim(ground_truth, prediction, mask):
    # Compute SSIM between two images only in the masked region
    # masked_ground_truth = ground_truth * mask
    # score_baseline = structural_similarity(masked_ground_truth, ground_truth,  data_range=np.max(ground_truth) - np.min(ground_truth))
    # ground_truth = ground_truth[np.where(mask == 0)]
    # prediction = prediction[np.where(mask == 0)]
    # score = structural_similarity(
    #     ground_truth,
    #     prediction,
    #     data_range=np.max(prediction) - np.min(prediction),
    #     mask=mask,
    # )
    mask = 1 - mask
    _, smap = structural_similarity(
        ground_truth,
        prediction,
        data_range=1,
        gaussian_weights=True,
        sigma=4,
        full=True,
    )
    smap_masked = np.multiply(smap, mask)
    return np.sum(smap_masked) / np.sum(mask)
    # normalized_score = (score-score_baseline)/(1-score_baseline)
    # return score


def calculate_psnr(ground_truth, prediction, mask):
    # Compute PSNR between two images only in the masked region
    ground_truth = ground_truth[np.where(mask == 0)]
    prediction = prediction[np.where(mask == 0)]
    mse = np.mean((ground_truth - prediction) ** 2)
    if mse == 0:
        return 100
    max_pixel = np.max(prediction)
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def calculate_rmse(ground_truth, prediction, mask):
    # Compute RMSE between two images only in the masked region
    ground_truth = ground_truth[np.where(mask == 0)]
    prediction = prediction[np.where(mask == 0)]
    rmse = np.sqrt(np.mean((ground_truth - prediction) ** 2))
    return rmse


def imshow_plots(model_path, counter_stop=10):
    gts = np.sort(glob(os.path.dirname(model_path) + "/results/gt_*.npy"))
    fakes = np.sort(glob(os.path.dirname(model_path) + "/results/img_*.npy"))
    masks = np.sort(glob(os.path.dirname(model_path) + "/results/mask_*.npy"))
    counter = 0
    for gt, fake, mask in zip(gts, fakes, masks):
        fake = np.load(fake)
        gt = np.load(gt)
        mask = np.ma.make_mask(np.load(mask)[:, :, 0])
        fig, ax = plt.subplots(1, 3, figsize=(9, 3), dpi=300, sharey=True)
        ax[0].imshow(fake, cmap="gray")
        ax[0].set_title("Inpainted", y=1.02)
        ax[1].imshow(gt, cmap="gray")
        ax[1].set_title("Ground Truth", y=1.02)
        img = ax[2].imshow((fake - gt), cmap="gray")
        ax[2].set_title("Difference", y=1.02)
        cax = fig.add_axes([0.92, 0.15, 0.01, 0.69])
        plt.colorbar(img, cax=cax)
        counter += 1
        if counter > counter_stop:
            break
