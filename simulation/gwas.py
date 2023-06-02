import statsmodels.api as sm
import numpy as np
import pandas as pd


def calculate_weights(sampling_data, probs):
    control_case_y = sampling_data[1]
    missing_w_mat = []
    for i, j in enumerate(probs):
        if j >= 0.5:
            missing_w_mat.insert(i, j)
        else:
            missing_w_mat.insert(i, 1 - j)

    missing_w_mat = np.array(missing_w_mat)
    control_case_w_mat = np.tile(1, len(control_case_y))

    w_mat = np.concatenate((control_case_w_mat, missing_w_mat))

    return w_mat

def calculate_gwas_y(sampling_data, probs):
    control_case_y = sampling_data[1]
    missing_gwas_y = []
    for i, j in enumerate(probs):
        if j >= 0.5:
            missing_gwas_y.insert(i, 1)
        else:
            missing_gwas_y.insert(i, 0)

    y_gwas = np.concatenate((np.array(control_case_y), np.array(missing_gwas_y)))

    #for case, y is 2. So, assign 1.
    y_gwas[y_gwas==2]=1

    return y_gwas

def sort_gwas_snp(sampling_data, is_missing=False):
    control_case_snp = sampling_data[0][:, -1]
    if is_missing:
        missing_snp = sampling_data[2][:, -1]

        snp_gwas = np.concatenate((np.array(control_case_snp), np.array(missing_snp)))
    else:
        snp_gwas = control_case_snp

    return snp_gwas

def weighted_gee(i, snp, y, w_mat):
    family = sm.families.Binomial()
    #link = sm.families.links.logit
    va = sm.cov_struct.Independence()
    group = np.arange(start=1, stop=y.shape[0]+1, step=1)


    res = pd.DataFrame(columns=['num_data','snp_idx', 'coef0','coef', 'std','p_value'])


    X = sm.cased_constant(snp)
    model = sm.GEE(endog=y, exog=X, groups=group,
                   family=family, cov_struct=va, weights=w_mat, missing='none')
    result = model.fit()
    p = result.pvalues[1]
    std = result.bse[1]
    coef = result.params[1]
    coef0 = result.params[0]

    res = res.append(pd.DataFrame([[len(y), i, coef0, coef, std, p]], columns=['num_data','snp_idx', 'coef0','coef','std', 'p_value']), ignore_index=True, sort=True)

    #res.to_csv(path+'wgee_res.csv', index=False)

    return res

def gee(i, snp, y):
    family = sm.families.Binomial()
    #link = sm.families.links.logit
    va = sm.cov_struct.Independence()
    group = np.arange(start=1, stop=y.shape[0]+1, step=1)


    res = pd.DataFrame(columns=['num_data', 'snp_idx', 'coef0','coef', 'std','p_value'])


    X = sm.cased_constant(snp)
    model = sm.GEE(endog=y, exog=X, groups=group,
                   family=family, cov_struct=va, missing='none')
    result = model.fit()
    p = result.pvalues[1]
    std = result.bse[1]
    coef = result.params[1]
    coef0 = result.params[0]

    res = res.append(pd.DataFrame([[len(y), i, coef0, coef, std, p]], columns=['num_data','snp_idx', 'coef0','coef','std', 'p_value']), ignore_index=True, sort=True)


    #res.to_csv(path+'wgee_res.csv', index=False)

    return res

def logistic(i, snp, y):
    #group = np.arange(start=1, stop=y.shape[0]+1, step=1)

    res = pd.DataFrame(columns=['num_data', 'snp_idx', 'coef0', 'coef', 'std','p_value'])


    X = sm.cased_constant(snp)
    model = sm.Logit(endog=y, exog=X, missing='none')
    result = model.fit(disp=0)
    p = result.pvalues[1]
    std = result.bse[1]
    coef = result.params[1]
    coef0 = result.params[0]

    res = res.append(pd.DataFrame([[len(y), i, coef0, coef, std, p]], columns=['num_data','snp_idx', 'coef0','coef','std', 'p_value']), ignore_index=True, sort=True)

    # res = res.append(pd.DataFrame([[len(y), i, coef, std, p]], columns=['num_data', 'snp_idx', 'coef', 'std', 'p_value']), ignore_index=True, sort=True)

    return res