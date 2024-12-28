import os
import sys
from glob import glob

if len(sys.argv) < 3:
    sys.exit()

models = glob(f"checkpoint/{sys.argv[1]}/*.pth")
models.sort()
for i, model in enumerate(models):
    print(model.split("/")[-1], f"{i}/{len(models)}")
    num_iters = int(model.split("_")[-1].split(".")[0])

    try:
        n = float(sys.argv[3])
    except:
        n = 2

    path = f"checkpoint/{sys.argv[1]}/finetune/{num_iters}+{int(num_iters//n)}/"
    if not os.path.exists(path + "g_final.pth"):
        cmd = f"python run.py --data_root data/SARS-COV-2-Ct-Scan-Dataset/resized/non-COVID/ --mask_root data/SARS-COV-2-Ct-Scan-Dataset/resized/masks/{sys.argv[2]}/ --finetune --model_path {model} --num_iters {int(num_iters + num_iters//n)} --model_save_path {path}"
        print(cmd)
        os.system(cmd)
