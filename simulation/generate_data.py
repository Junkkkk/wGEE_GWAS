from scipy.special import logit, expit
from scipy.stats import norm
import random
import numpy as np
import itertools


def calculate_beta(maf, coef, h_i, kappa, nu):
    sex_beta = coef[:, 0]
    var_sex = 1 * 0.5 * 0.5

    mri_betas = coef[:, 1:]
    var_mri = 1

    var_snp = 2 * maf * (1 - maf)

    var_e = 1

    if nu==1:
        snp2ad_beta =0

        a = np.sum((mri_betas**2)) * var_snp * (h_i - 1)
        b = -h_i * (sex_beta ** 2 * var_sex + np.sum((mri_betas ** 2) * var_mri) + var_e)

        snp2mri_beta= np.sqrt((b/a)[0])

    else:
        h_d = kappa * h_i
        a = var_snp * (h_d - 1)
        b = h_d * np.sum((mri_betas ** 2)) * var_snp
        c = -h_d * (sex_beta ** 2 * var_sex + np.sum((mri_betas ** 2) * var_mri) + var_e)

        a_p = var_snp * h_i
        b_p = (h_i - 1) * np.sum((mri_betas ** 2)) * var_snp
        c_p = -h_i * (sex_beta ** 2 * var_sex + np.sum((mri_betas ** 2) * var_mri) + var_e)

        A = np.array([[a, b], [a_p, b_p]])
        B = np.array([c, c_p])
        C = np.linalg.solve(A, B)

        snp2ad_beta = np.sqrt(C)[0][0]
        snp2mri_beta = np.sqrt(C)[1][0]

    return snp2ad_beta, snp2mri_beta

def generate_snps(N, maf, snp_seed):
    np.random.seed(snp_seed)
    return np.random.binomial(2,maf,N)


def generate_features_snp2mri(N, coef, snps, snp2mri_beta, nu, K, seed):
    np.random.seed(seed)
    dim_x = coef.shape[1]
    x = np.zeros((N, dim_x+1))

    for i in range(x.shape[1]):
        #Sex
        if i == 0:
            x[:, i] = np.random.binomial(1, 0.5, N)
        #only MRI traits 55 not log ICV & cognitive trait
        elif i < x.shape[1]-1:
            if i <= K:
                x[:, i] = np.random.normal(0, 1, N) - nu*snp2mri_beta*snps
            else:
                x[:, i] = np.random.normal(0, 1, N)
        else:
            x[:, i] = snps

    return x

def generate_snp2mri_snp2ad_phenotypes(x, coef, snp2ad_beta, nu):
    #snp2mri -> 0
    #snp2mri_beta = np.expand_dims(np.array([0]), axis=0)
    #snp2ad
    snp2ad_beta = np.expand_dims(np.array([(1-nu)*snp2ad_beta]), axis=0)

    coef = np.concatenate([coef, snp2ad_beta], axis=1)
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

def assign_diagnosis_cn_ad_binomial(y,N,seed):
    np.random.seed(seed)
    pheno = np.array(np.random.binomial(1, y, N))
    pheno[pheno==1]=2

    cn_index = list(filter(lambda x: pheno[x] == 0, range(len(pheno))))
    ad_index = list(filter(lambda x: pheno[x] == 2, range(len(pheno))))

    pheno = list(pheno)

    return pheno, [cn_index, ad_index]

def sampling_data(n,x,y,index,seed):
    random.seed(seed)

    cn_index=random.sample(index[0], k=n)
    mci_index=random.sample(index[1], k=n)
    ad_index=random.sample(index[2], k=n)

    cn_ad_index = cn_index + ad_index

    cn_ad_x = x[cn_ad_index]
    mci_x = x[mci_index]

    cn_ad_y = np.array(y)[cn_ad_index]
    mci_y = np.array(y)[mci_index]

    return [cn_ad_x, cn_ad_y, mci_x, mci_y]

def sampling_data_cn_ad_considering_weights(n,n_mci,x,y,prob,index,seed):
    np.random.seed(seed)
    prob=np.array(prob)
    cn_prob=prob[index[0]]
    cn_prob=list(1-cn_prob)
    cn_prob=cn_prob/np.sum(cn_prob)
    
    ad_prob=prob[index[1]]
    ad_prob=ad_prob/np.sum(ad_prob)


    cn_index=np.random.choice(index[0], n, p=list(cn_prob), replace=False)
    index_not_cn= [i for i in index[0] if i not in cn_index]
    mci_index1=np.random.choice(index_not_cn, int(n_mci/2), replace=False)
    ad_index=np.random.choice(index[1], n, p=list(ad_prob), replace=False)
    index_not_ad= [i for i in index[1] if i not in ad_index]
    mci_index2=np.random.choice(index_not_ad, int(n_mci/2), replace=False)

    mci_index = list(mci_index1)+list(mci_index2)
    cn_ad_index = list(cn_index)+list(ad_index)

    cn_ad_x = x[cn_ad_index]
    mci_x = x[mci_index]

    cn_ad_y = np.array(y)[cn_ad_index]
    mci_y = np.array(y)[mci_index]

    return [cn_ad_x, cn_ad_y, mci_x, mci_y]

