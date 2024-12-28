## **Data Preprocessing**

The image files were all resized to 256x256 pixels, converted to single channel (`'L'`) arrays, and saved as NumPy Arrays (`.npy`). The masks were created as mentioned in the report. Refer `create-dataset.ipynb`. Relavent functions are in `utilities.py`.

## **Training the model**

Following is a sample comamnd for training the model:

```
python3 run.py --data_root data/SARS-COV-2-Ct-Scan-Dataset/resized/non-COVID/ --mask_root data/SARS-COV-2-Ct-Scan-Dataset/resized/masks/2_HVstrips/ --num_iters 50000 --model_save_path checkpoint/model6/
```

## **Fine-tuning**

Following is a sample command for finetuning the model:

```
python finetune.py model6 2_HVstrips 2
```

## **Generating Inpainted Results**

The inpainted images are generated as NumPy arrays along with the ground truth and mask arrays. Following is a sample command:

```
python create-results.py model6 2_HVstrips
```

## **Final Plots**

The `imshow-plots.ipynb` notebook contains the `plt.imshow` plots and code that compare the inpainted result with ground truth and difference arrays. 

`metrics.ipynb` contains the plots and code for generating and plotting the metrics (`SSIM`, `PSNR`, `RMSE`).

All the plots are stored in `results/` under relavent subfolders `model1/`,`model2/`, etc.