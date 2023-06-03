# wGEE_GWAS
Simulation cods for weighted GWAS by imputing phenotypes based on machine-learning.


# Requirement
- python >= 3.8.0
- numpy >= 1.24.2
- pandas >= 1.5.3
- scikit-learn >= 1.1.1
- statsmodels >= 0.13.2

# Installation
```bash
$ git clone https://github.com/Junkkkk/wGEE_GWAS.git
$ cd ./wGEE_GWAS
$ pip install -r requirements.txt
```

# Code

```bash
python train.py -h
usage: train.py [-h] [--save_path SAVE_PATH] [--N N] [--n_case N_CASE] [--n_control N_CONTROL] [--n_missing N_MISSING]
[--n_predictor N_PREDICTOR] [--maf MAF] [--h_i H_I] [--h_d H_D] [--nu NU] [--K K]

optional arguments:
  -h, --help            show this help message and exit
  --save_path SAVE_PATH
                        (default=./result)
  --N N  # total population size
                        (default=50000)
  --n_case N_CASE  # the number of case samples
                        (default=5000)
  --n_control N_CONTROL  # the number of control samples
                        (default=5000)
  --n_missing N_MISSING  # the number of missing samples
                        (default=5000)
  --n_predictor N_PREDICTOR  # the number of total predictors to predict the disease model
                        (default=5000)
  --maf MAF  # minor allele frequency
                        (default=0.1)
  --h_i H_I  # ratio of indirect effect variance among total variance
                        (default=0.001)
  --h_d H_D  # ratio of direct effect variance among total variance
                        (default=0.0001)
  --nu NU  # ratio of indirect effect size of SNP on disease status (0~1)
                        (default=0.5)
  --k NU  # ratio of indirect effect size of SNP on disease status (0~1)
                      (default=0.5)
```
# Test Code
```bash
python test.py -h
usage: test.py [-h] [--folder FOLDER] [--data_path DATA_PATH] [--save_path SAVE_PATH] [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS] [--gpu_num GPU_NUM] [--model_num MODEL_NUM]
               [--model_name MODEL_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --folder FOLDER       (default=0)
  --data_path DATA_PATH
                        (default=./data)
  --save_path SAVE_PATH
                        (default=./result)
  --batch_size BATCH_SIZE
                        (default=1)
  --num_workers NUM_WORKERS
                        (default=8)
  --gpu_num GPU_NUM     (default=0)
  --model_num MODEL_NUM
                        0: efficinetnet-b0, 1: efficinetnet-b1, 2: efficinetnet-b2, 3: efficinetnet-b3, 4: efficinetnet-b4, 5: efficinetnet-b5, 6: vit, 7: cait, 8: deepvit, 9: resnet50,
                        10: resnet101, 11: resnet152, 12: densenet121, 13: densenet161, 14: densenet169, 15: densenet201, (default=0, efficinetnet-b0)
  --model_name MODEL_NAME
                        (default=None)
```


