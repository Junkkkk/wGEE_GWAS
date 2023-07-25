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
python main.py -h
usage: main.py [-h] [--save_path SAVE_PATH] [--N N] [--n_case N_CASE] [--n_control N_CONTROL] [--n_missing N_MISSING]
[--n_predictor N_PREDICTOR] [--maf MAF] [--h_i H_I] [--h_d H_D] [--nu NU] [--k K]

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
  --k K  # the number of predictors affected by SNP (K<= n_predictor)
                        (default=10)
```
# How to generate data
$\gamma_j, \alpha \sim U(-1,1) $.
<br/>
$𝑝_𝑖 =logit^{−1}{[𝐷_𝑖α+∑_{𝑗=1}^{k}{(𝑋_{𝑖𝑗}^{'}-{\nu}G_i)γ_𝑗} + ∑_{𝑗=k+1}^{n}{𝑋_{𝑖𝑗}^{'}γ_𝑗} +(1−{\nu})𝐺_𝑖𝛽_𝐷 + 𝜖_𝑖]} $.
<br/>
$y_i \sim Bernoulli(p_i)$.
<br/> <br/>

![Generate_data](https://github.com/Junkkkk/wGEE_GWAS/assets/46311404/8f0fd174-29c1-4783-bb68-19b0c447fbc4)

# Output
- output : csv file 
- Example

|SNP idx|coef0|coef|std|p_value|
|:---:|:---:|:---:|:---:|:---:|
|1|0.0085|-0.4123|0.0483|1.57e-17|
|2|-0.0049|0.0213|0.0483|0.6059|
|3|-0.0009|0.0459|0.0483|0.3201|

- column
  - SNP idx : SNP index
  - coef0 : constant coefficient
  - coef : SNP coefficient
  - std : standard deviation of SNP coefficient
  - p_value : p value of SNP coefficient
