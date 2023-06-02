from scipy.special import logit, expit
from scipy.stats import norm
import random
import numpy as np
import itertools


def calculate_beta(maf, coef, h_i, h_d, nu):
    sex_beta = coef[:, 0]
    var_sex = 1 * 0.5 * 0.5

    predictor_betas = coef[:, 1:]
    var_predictor = 1

    var_snp = 2 * maf * (1 - maf)

    var_e = 1

    if nu==1:
        snp2disease_beta =0

        a = np.sum((predictor_betas**2)) * var_snp * (h_i - 1)
        b = -h_i * (sex_beta ** 2 * var_sex + np.sum((predictor_betas ** 2) * var_predictor) + var_e)

        snp2predictor_beta= np.sqrt((b/a)[0])

    else:
        a = var_snp * (h_d - 1)
        b = h_d * np.sum((predictor_betas ** 2)) * var_snp
        c = -h_d * (sex_beta ** 2 * var_sex + np.sum((predictor_betas ** 2) * var_predictor) + var_e)

        a_p = var_snp * h_i
        b_p = (h_i - 1) * np.sum((predictor_betas ** 2)) * var_snp
        c_p = -h_i * (sex_beta ** 2 * var_sex + np.sum((predictor_betas ** 2) * var_predictor) + var_e)

        A = np.array([[a, b], [a_p, b_p]])
        B = np.array([c, c_p])
        C = np.linalg.solve(A, B)

        snp2disease_beta = np.sqrt(C)[0][0]
        snp2predictor_beta = np.sqrt(C)[1][0]

    return snp2disease_beta, snp2predictor_beta

def generate_snps(N, maf, snp_seed):
    np.random.seed(snp_seed)
    return np.random.binomial(2,maf,N)


def generate_features_snp2predictor(N, coef, snps, snp2predictor_beta, nu, K, seed):
    np.random.seed(seed)
    dim_x = coef.shape[1]
    x = np.zeros((N, dim_x+1))

    for i in range(x.shape[1]):
        #Sex
        if i == 0:
            x[:, i] = np.random.binomial(1, 0.5, N)
        #Predictors
        elif i < x.shape[1]-1:
            if i <= K:
                x[:, i] = np.random.normal(0, 1, N) - nu*snp2predictor_beta*snps
            else:
                x[:, i] = np.random.normal(0, 1, N)
        #SNP
        else:
            x[:, i] = snps

    return x

def generate_snp2predictor_snp2disease_phenotypes(x, coef, snp2disease_beta, nu):
    snp2disease_beta = np.expand_dims(np.array([(1-nu)*snp2disease_beta]), axis=0)

    coef = np.concatenate([coef, snp2disease_beta], axis=1)
    y_hat = np.matmul(x, coef.T)
    y_hat_prob = expit(y_hat)

    y = list(itertools.chain(*y_hat_prob))

    return y


def sort_x_y(x,y):
    ordered = np.argsort(y)
    ordered = ordered[::-1]

    y=[y[i] for i in ordered]
    x=x[ordered]

    return x, y

def assign_diagnosis_control_case_binomial(y,N,seed):
    np.random.seed(seed)
    pheno = np.array(np.random.binomial(1, y, N))
    pheno[pheno==1]=2

    control_index = list(filter(lambda x: pheno[x] == 0, range(len(pheno))))
    case_index = list(filter(lambda x: pheno[x] == 2, range(len(pheno))))

    pheno = list(pheno)

    return pheno, [control_index, case_index]

def sampling_data(n,x,y,index,seed):
    random.seed(seed)

    control_index=random.sample(index[0], k=n)
    missing_index=random.sample(index[1], k=n)
    case_index=random.sample(index[2], k=n)

    control_case_index = control_index + case_index

    control_case_x = x[control_case_index]
    missing_x = x[missing_index]

    control_case_y = np.array(y)[control_case_index]
    missing_y = np.array(y)[missing_index]

    return [control_case_x, control_case_y, missing_x, missing_y]

def sampling_data_control_case_considering_weights(n_case, n_control,n_missing,x,y,prob,index,seed):
    np.random.seed(seed)
    prob=np.array(prob)
    control_prob=prob[index[0]]
    control_prob=list(1-control_prob)
    control_prob=control_prob/np.sum(control_prob)
    
    case_prob=prob[index[1]]
    case_prob=case_prob/np.sum(case_prob)


    control_index=np.random.choice(index[0], n_control, p=list(control_prob), replace=False)
    index_not_control= [i for i in index[0] if i not in control_index]
    missing_index1=np.random.choice(index_not_control, int(n_missing/2), replace=False)
    case_index=np.random.choice(index[1], n_case, p=list(case_prob), replace=False)
    index_not_case= [i for i in index[1] if i not in case_index]
    missing_index2=np.random.choice(index_not_case, int(n_missing/2), replace=False)

    missing_index = list(missing_index1)+list(missing_index2)
    control_case_index = list(control_index)+list(case_index)

    control_case_x = x[control_case_index]
    missing_x = x[missing_index]

    control_case_y = np.array(y)[control_case_index]
    missing_y = np.array(y)[missing_index]

    return [control_case_x, control_case_y, missing_x, missing_y]

