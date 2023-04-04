from generate_data import *
from gwas import *
from pred_model import *

import os
import random
import numpy as np
import pandas as pd

def simulation(config, coef, i, seed, snp_seed):
    N=config.N
    n=config.n
    n_mci=config.n_mci

    maf=config.maf
    path=config.save_path
    snp2mri_ratio = config.snp2mri_ratio
    h_i = config.h_i
    kappa = config.kappa
    nu = config.nu
    K = config.K


    '''
    :param N: population size
    :param n: simulation sample size
    :param coef: coeffieient for generating x
    :param maf: maf
    :param path: wgee save path
    :param i: simulation index
    '''

    #generate input features and phenotype for prediction
    snps = generate_snps(N, maf, snp_seed)
    
    # logistic coefficient from real data
    coef = coef[:, 1:57]

    if ((nu==0)|(nu==1)) & (kappa==0):
        snp2ad_beta=0; snp2mri_beta=0
    else:
        snp2ad_beta, snp2mri_beta = calculate_beta(maf, coef, h_i, kappa, nu)

    # both direct and indirect path
    x=generate_features_snp2mri(N, coef, snps, snp2mri_beta, nu, K, seed)
    y=generate_snp2mri_snp2ad_phenotypes(x, coef, snp2ad_beta, nu)


    #just assign x & y -> sort(logit(y))
    x,y = sort_x_y(x,y)
    pheno, diagnosis_index=assign_diagnosis_cn_ad_binomial(y,N,seed)

    #print('simulation {}'.format(i))
    samplingdata=sampling_data_cn_ad_considering_weights(n=n, n_mci=n_mci, x=x, y=pheno, prob=y, index=diagnosis_index, seed=seed)

    split_data = split_train_test(samplingdata)
    model=train_model(split_data)
    auc, acc, sen, spe = evaluate_model(model, split_data)
    mci_probs = predict_mci(model, samplingdata)
    mci_auc, mci_acc, mci_sen, mci_spe = evaluate_mci(model, samplingdata)

    # for gwas
    weight_mat = calculate_weights(samplingdata, mci_probs)
    gwas_y_mci = calculate_gwas_y(samplingdata, mci_probs)
    gwas_snp_mci = sort_gwas_snp(samplingdata, is_mci=True)

    os.makedirs(path, exist_ok=True)
    os.makedirs(path + '{}_snp2mri_{}_snp2ad_mri_K_{}_simultaneous_N_mci_{}'.format(str(nu),str(np.round((1-nu),1)),str(K),str(n_mci)), exist_ok=True)

    # with mci
    with open(os.path.join(path,
                           '{}_snp2mri_{}_snp2ad_mri_K_{}_simultaneous_N_mci_{}/prediction_result_h_i_{}_kappa_{}_snp2mri_beta_{}_snp2ad_beta_{}_maf_{}_N_mci_{}.txt'.format(str(nu),str(np.round((1-nu),1)), str(K),str(n_mci),h_i, kappa,
                                                                                                                   str(np.round(snp2mri_beta,4)),
                                                                                                                   str(np.round(snp2ad_beta,4)),
                                                                                                                   maf, str(n_mci))),
              "a") as f:
        f.write('{} {} {} {} {} {} {} {} {}'.format(i, auc, acc, sen, spe, mci_auc, mci_acc, mci_sen, mci_spe) + '\n')

    # wgee with  MCI
    wgee_res = weighted_gee(i=i, snp=gwas_snp_mci, y=gwas_y_mci, w_mat=weight_mat)

    # os.makedirs(path + '{}_snp2mri'.format(snp2mri_ratio, h_i, kappa), exist_ok=True)

    if not os.path.exists(
            path + '{}_snp2mri_{}_snp2ad_mri_K_{}_simultaneous_N_mci_{}/wgee_h_i_{}_kappa_{}_snp2mri_beta_{}_snp2ad_beta_{}_maf_{}_N_mci_{}.csv'.format(str(nu),str(np.round((1-nu),1)), str(K), str(n_mci), h_i, kappa, str(np.round(snp2mri_beta,4)),
                                                                                                  str(np.round(snp2ad_beta,4)), maf, str(n_mci))):
        wgee_res.to_csv(
            path + '{}_snp2mri_{}_snp2ad_mri_K_{}_simultaneous_N_mci_{}/wgee_h_i_{}_kappa_{}_snp2mri_beta_{}_snp2ad_beta_{}_maf_{}_N_mci_{}.csv'.format(str(nu),str(np.round((1-nu),1)),str(K),str(n_mci),h_i, kappa, str(np.round(snp2mri_beta,4)),
                                                                                                  str(np.round(snp2ad_beta,4)), maf, str(n_mci)),
            index=False, mode='w')
    else:
        wgee_res.to_csv(
            path + '{}_snp2mri_{}_snp2ad_mri_K_{}_simultaneous_N_mci_{}/wgee_h_i_{}_kappa_{}_snp2mri_beta_{}_snp2ad_beta_{}_maf_{}_N_mci_{}.csv'.format(str(nu),str(np.round((1-nu),1)),str(K),str(n_mci),h_i, kappa, str(np.round(snp2mri_beta,4)),
                                                                                                  str(np.round(snp2ad_beta,4)), maf, str(n_mci)),
            index=False, mode='a', header=False)

    # gee with MCI

    gee_res_mci = gee(i=i, snp=gwas_snp_mci, y=gwas_y_mci)

    if not os.path.exists(
            path + '{}_snp2mri_{}_snp2ad_mri_K_{}_simultaneous_N_mci_{}/with_mci_gee_h_i_{}_kappa_{}_snp2mri_beta_{}_snp2ad_beta_{}_maf_{}_N_mci_{}.csv'.format(str(nu),str(np.round((1-nu),1)),str(K), str(n_mci),h_i, kappa,
                                                                                                              str(np.round(snp2mri_beta,4)),
                                                                                                              str(np.round(snp2ad_beta,4)), maf, str(n_mci))):
        gee_res_mci.to_csv(
            path + '{}_snp2mri_{}_snp2ad_mri_K_{}_simultaneous_N_mci_{}/with_mci_gee_h_i_{}_kappa_{}_snp2mri_beta_{}_snp2ad_beta_{}_maf_{}_N_mci_{}.csv'.format(str(nu),str(np.round((1-nu),1)),str(K),str(n_mci), h_i, kappa,
                                                                                                              str(np.round(snp2mri_beta,4)),
                                                                                                              str(np.round(snp2ad_beta,4)),
                                                                                                              maf, str(n_mci)),
            index=False,
            mode='w')
    else:
        gee_res_mci.to_csv(
            path + '{}_snp2mri_{}_snp2ad_mri_K_{}_simultaneous_N_mci_{}/with_mci_gee_h_i_{}_kappa_{}_snp2mri_beta_{}_snp2ad_beta_{}_maf_{}_N_mci_{}.csv'.format(str(nu),str(np.round((1-nu),1)), str(K),str(n_mci),h_i, kappa,
                                                                                                              str(np.round(snp2mri_beta,4)),
                                                                                                              str(np.round(snp2ad_beta,4)),
                                                                                                              maf, str(n_mci)),
            index=False,
            mode='a', header=False)

    #Logistic without MCI...
    gwas_y = np.array(samplingdata[1])
    gwas_y[gwas_y==2] = 1
    gwas_snp=sort_gwas_snp(samplingdata, is_mci=False)


    log_res = logistic(i=i, snp=gwas_snp, y=gwas_y)

    if not os.path.exists(
            path + '{}_snp2mri_{}_snp2ad_mri_K_{}_simultaneous_N_mci_{}/no_mci_log_h_i_{}_kappa_{}_snp2mri_beta_{}_snp2ad_beta_{}_maf_{}_N_mci_{}.csv'.format(str(nu),str(np.round((1-nu),1)),str(K), str(n_mci),h_i, kappa,
                                                                                                              str(np.round(snp2mri_beta,4)),
                                                                                                              str(np.round(snp2ad_beta,4)), maf, str(n_mci))):
        log_res.to_csv(
            path + '{}_snp2mri_{}_snp2ad_mri_K_{}_simultaneous_N_mci_{}/no_mci_log_h_i_{}_kappa_{}_snp2mri_beta_{}_snp2ad_beta_{}_maf_{}_N_mci_{}.csv'.format(str(nu),str(np.round((1-nu),1)),str(K),str(n_mci), h_i, kappa,
                                                                                                              str(np.round(snp2mri_beta,4)),
                                                                                                              str(np.round(snp2ad_beta,4)),
                                                                                                              maf, str(n_mci)),
            index=False,
            mode='w')
    else:
        log_res.to_csv(
            path + '{}_snp2mri_{}_snp2ad_mri_K_{}_simultaneous_N_mci_{}/no_mci_log_h_i_{}_kappa_{}_snp2mri_beta_{}_snp2ad_beta_{}_maf_{}_N_mci_{}.csv'.format(str(nu),str(np.round((1-nu),1)), str(K),str(n_mci),h_i, kappa,
                                                                                                              str(np.round(snp2mri_beta,4)),
                                                                                                              str(np.round(snp2ad_beta,4)),
                                                                                                              maf, str(n_mci)),
            index=False,
            mode='a', header=False)