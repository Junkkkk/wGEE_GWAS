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
                        (default=50)
  --maf MAF  # minor allele frequency
                        (default=0.1)
  --h_i H_I  # ratio of indirect effect variance among total variance
                        (default=0.001)
  --h_d H_D  # ratio of direct effect variance among total variance
                        (default=0.0001)
  --nu NU  # ratio of indirect effect size of SNP on disease status (0~1)
                        (default=0.5)
  --k NU  # the number of predictors affected by SNP (K<= n_predictor)
                        (default=10)
```
# How to generate data
$\gamma_j, \alpha \sim U(-1,1) $.
<br/>
$𝑝_𝑖 =logit^{−1}{𝐷_𝑖α + ∑_{𝑗=1}^{k}{(𝑋_𝑖𝑗-{\nu}G_iγ_𝑗} + ∑_{𝑗=k+1}^{n}{𝑋_𝑖𝑗γ_𝑗} +(1−\nu)𝐺_𝑖𝛽_𝐷 + 𝜖_𝑖} $.
<br/>
$y_i \sim Bernoulli(p_i)$.
<br/> <br/>


