import os
import sys
from glob import glob

if len(sys.argv) < 2:
    sys.exit()

model_paths = glob(f"checkpoint/{sys.argv[1]}/finetune/**/g_final.pth", recursive=True)
data_root = "data/SARS-COV-2-Ct-Scan-Dataset/resized/non-COVID/"
mask_root = os.path.dirname(os.path.dirname(data_root)) + f"/masks/{sys.argv[2]}/"
for model_path in model_paths:
    try:
        result_path = (
            str(sys.argv[3]) + "/" + os.path.basename(os.path.dirname(model_path)) + "/"
        )
    except:
        result_path = os.path.dirname(model_path)
    test_cmd = f"python run.py --data_root {data_root} --mask_root {mask_root} --result_save_path {result_path} --model_path {model_path} --test"
    print(test_cmd)
    os.system(test_cmd)
