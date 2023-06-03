from generate_data import *
from gwas import *
from pred_model import *

import os
import random
import numpy as np
import pandas as pd

def simulation(config, i, seed, snp_seed):
    N=config.N
    n_case=config.n_case
    n_control=config.n_control
    n_missing=config.n_missing
    n_predictor=config.n_predictor

    maf=config.maf
    path=config.save_path
    h_i = config.h_i
    h_d = config.h_d

    nu = config.nu
    K = config.k

    # generate input features and phenotype for prediction
    snps = generate_snps(N, maf, snp_seed)
    
    # coeffieient for generating predictor from uniform distribution
    np.random.seed(seed)
    coef = np.random.uniform (-1, 1, size=n_predictor+1)

    if ((nu==0)|(nu==1)) & (h_i==0):
        snp2disease_beta=0; snp2predictor_beta=0
    else:
        snp2disease_beta, snp2predictor_beta = calculate_beta(maf, coef, h_i, kappa, nu)

    # both direct and indirect path
    x=generate_features_snp2predictor(N, coef, snps, snp2predictor_beta, nu, K, seed)
    y=generate_snp2predictor_snp2disease_phenotypes(x, coef, snp2disease_beta, nu)


    #just assign x & y -> sort(logit(y))
    x,y = sort_x_y(x,y)
    pheno, diagnosis_index=assign_diagnosis_control_case_binomial(y,N,seed)

    #print('simulation {}'.format(i))
    samplingdata=sampling_data_control_case_considering_weights(n_case=n_case, n_control=n_control, n_missing=n_missing, x=x, y=pheno, prob=y, index=diagnosis_index, seed=seed)

    split_data = split_train_test(samplingdata)
    model=train_model(split_data)
    auc, acc, sen, spe = evaluate_model(model, split_data)
    missing_probs = predict_missing(model, samplingdata)
    missing_auc, missing_acc, missing_sen, missing_spe = evaluate_missing(model, samplingdata)

    # for gwas
    weight_mat = calculate_weights(samplingdata, missing_probs)
    gwas_y_missing = calculate_gwas_y(samplingdata, missing_probs)
    gwas_snp_missing = sort_gwas_snp(samplingdata, is_missing=True)

    os.makedirs(path, exist_ok=True)
    os.makedirs(path + '{}_snp2predictor_{}_snp2disease_predictor_K_{}_N_missing_{}'.format(str(nu),str(np.round((1-nu),1)),str(K),str(n_missing)), exist_ok=True)

    # with missing
    # prediction performance for missing group
    with open(os.path.join(path,
                           '{}_snp2predictor_{}_snp2disease_predictor_K_{}_N_missing_{}/prediction_result_h_i_{}_kappa_{}_snp2predictor_beta_{}_snp2disease_beta_{}_maf_{}_N_missing_{}.txt'.format(str(nu),str(np.round((1-nu),1)), str(K),str(n_missing),h_i, kappa,
                                                                                                                   str(np.round(snp2predictor_beta,4)),
                                                                                                                   str(np.round(snp2disease_beta,4)),
                                                                                                                   maf, str(n_missing))),
              "a") as f:
        f.write('{} {} {} {} {} {} {} {} {}'.format(i, auc, acc, sen, spe, missing_auc, missing_acc, missing_sen, missing_spe) + '\n')

    # wgee with  missing
    wgee_res = weighted_gee(i=i, snp=gwas_snp_missing, y=gwas_y_missing, w_mat=weight_mat)

    if not os.path.exists(
            path + '{}_snp2predictor_{}_snp2disease_predictor_K_{}_N_missing_{}/wgee_h_i_{}_kappa_{}_snp2predictor_beta_{}_snp2disease_beta_{}_maf_{}_N_missing_{}.csv'.format(str(nu),str(np.round((1-nu),1)), str(K), str(n_missing), h_i, kappa, str(np.round(snp2predictor_beta,4)),
                                                                                                  str(np.round(snp2disease_beta,4)), maf, str(n_missing))):
        wgee_res.to_csv(
            path + '{}_snp2predictor_{}_snp2disease_predictor_K_{}_N_missing_{}/wgee_h_i_{}_kappa_{}_snp2predictor_beta_{}_snp2disease_beta_{}_maf_{}_N_missing_{}.csv'.format(str(nu),str(np.round((1-nu),1)),str(K),str(n_missing),h_i, kappa, str(np.round(snp2predictor_beta,4)),
                                                                                                  str(np.round(snp2disease_beta,4)), maf, str(n_missing)),
            index=False, mode='w')
    else:
        wgee_res.to_csv(
            path + '{}_snp2predictor_{}_snp2disease_predictor_K_{}_N_missing_{}/wgee_h_i_{}_kappa_{}_snp2predictor_beta_{}_snp2disease_beta_{}_maf_{}_N_missing_{}.csv'.format(str(nu),str(np.round((1-nu),1)),str(K),str(n_missing),h_i, kappa, str(np.round(snp2predictor_beta,4)),
                                                                                                  str(np.round(snp2disease_beta,4)), maf, str(n_missing)),
            index=False, mode='a', header=False)

    # gee with missing

    gee_res_missing = gee(i=i, snp=gwas_snp_missing, y=gwas_y_missing)

    if not os.path.exists(
            path + '{}_snp2predictor_{}_snp2disease_predictor_K_{}_N_missing_{}/with_missing_gee_h_i_{}_kappa_{}_snp2predictor_beta_{}_snp2disease_beta_{}_maf_{}_N_missing_{}.csv'.format(str(nu),str(np.round((1-nu),1)),str(K), str(n_missing),h_i, kappa,
                                                                                                              str(np.round(snp2predictor_beta,4)),
                                                                                                              str(np.round(snp2disease_beta,4)), maf, str(n_missing))):
        gee_res_missing.to_csv(
            path + '{}_snp2predictor_{}_snp2disease_predictor_K_{}_N_missing_{}/with_missing_gee_h_i_{}_kappa_{}_snp2predictor_beta_{}_snp2disease_beta_{}_maf_{}_N_missing_{}.csv'.format(str(nu),str(np.round((1-nu),1)),str(K),str(n_missing), h_i, kappa,
                                                                                                              str(np.round(snp2predictor_beta,4)),
                                                                                                              str(np.round(snp2disease_beta,4)),
                                                                                                              maf, str(n_missing)),
            index=False,
            mode='w')
    else:
        gee_res_missing.to_csv(
            path + '{}_snp2predictor_{}_snp2disease_predictor_K_{}_N_missing_{}/with_missing_gee_h_i_{}_kappa_{}_snp2predictor_beta_{}_snp2disease_beta_{}_maf_{}_N_missing_{}.csv'.format(str(nu),str(np.round((1-nu),1)), str(K),str(n_missing),h_i, kappa,
                                                                                                              str(np.round(snp2predictor_beta,4)),
                                                                                                              str(np.round(snp2disease_beta,4)),
                                                                                                              maf, str(n_missing)),
            index=False,
            mode='a', header=False)

    #Logistic without missing...
    gwas_y = np.array(samplingdata[1])
    gwas_y[gwas_y==2] = 1
    gwas_snp=sort_gwas_snp(samplingdata, is_missing=False)


    log_res = logistic(i=i, snp=gwas_snp, y=gwas_y)

    if not os.path.exists(
            path + '{}_snp2predictor_{}_snp2disease_predictor_K_{}_N_missing_{}/without_missing_log_h_i_{}_kappa_{}_snp2predictor_beta_{}_snp2disease_beta_{}_maf_{}_N_missing_{}.csv'.format(str(nu),str(np.round((1-nu),1)),str(K), str(n_missing),h_i, kappa,
                                                                                                              str(np.round(snp2predictor_beta,4)),
                                                                                                              str(np.round(snp2disease_beta,4)), maf, str(n_missing))):
        log_res.to_csv(
            path + '{}_snp2predictor_{}_snp2disease_predictor_K_{}_N_missing_{}/without_missing_log_h_i_{}_kappa_{}_snp2predictor_beta_{}_snp2disease_beta_{}_maf_{}_N_missing_{}.csv'.format(str(nu),str(np.round((1-nu),1)),str(K),str(n_missing), h_i, kappa,
                                                                                                              str(np.round(snp2predictor_beta,4)),
                                                                                                              str(np.round(snp2disease_beta,4)),
                                                                                                              maf, str(n_missing)),
            index=False,
            mode='w')
    else:
        log_res.to_csv(
            path + '{}_snp2predictor_{}_snp2disease_predictor_K_{}_N_missing_{}/without_missing_log_h_i_{}_kappa_{}_snp2predictor_beta_{}_snp2disease_beta_{}_maf_{}_N_missing_{}.csv'.format(str(nu),str(np.round((1-nu),1)), str(K),str(n_missing),h_i, kappa,
                                                                                                              str(np.round(snp2predictor_beta,4)),
                                                                                                              str(np.round(snp2disease_beta,4)),
                                                                                                              maf, str(n_missing)),
            index=False,
            mode='a', header=False)