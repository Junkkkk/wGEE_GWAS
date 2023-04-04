import statsmodels.api as sm
import numpy as np
import pandas as pd


def calculate_weights(sampling_data, probs):
    cn_ad_y = sampling_data[1]
    mci_w_mat = []
    for i, j in enumerate(probs):
        if j >= 0.5:
            mci_w_mat.insert(i, j)
        else:
            mci_w_mat.insert(i, 1 - j)

    mci_w_mat = np.array(mci_w_mat)
    cn_ad_w_mat = np.tile(1, len(cn_ad_y))

    w_mat = np.concatenate((cn_ad_w_mat, mci_w_mat))

    return w_mat

def calculate_gwas_y(sampling_data, probs):
    cn_ad_y = sampling_data[1]
    mci_gwas_y = []
    for i, j in enumerate(probs):
        if j >= 0.5:
            mci_gwas_y.insert(i, 1)
        else:
            mci_gwas_y.insert(i, 0)

    y_gwas = np.concatenate((np.array(cn_ad_y), np.array(mci_gwas_y)))

    #for AD, y is 2. So, assign 1.
    y_gwas[y_gwas==2]=1

    return y_gwas

def sort_gwas_snp(sampling_data, is_mci=False):
    cn_ad_snp = sampling_data[0][:, -1]
    if is_mci:
        mci_snp = sampling_data[2][:, -1]

        snp_gwas = np.concatenate((np.array(cn_ad_snp), np.array(mci_snp)))
    else:
        snp_gwas = cn_ad_snp

    return snp_gwas

def weighted_gee(i, snp, y, w_mat):
    family = sm.families.Binomial()
    #link = sm.families.links.logit
    va = sm.cov_struct.Independence()
    group = np.arange(start=1, stop=y.shape[0]+1, step=1)


    res = pd.DataFrame(columns=['num_data','snp_idx', 'coef0','coef', 'std','p_value'])


    X = sm.add_constant(snp)
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


    X = sm.add_constant(snp)
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


    X = sm.add_constant(snp)
    model = sm.Logit(endog=y, exog=X, missing='none')
    result = model.fit(disp=0)
    p = result.pvalues[1]
    std = result.bse[1]
    coef = result.params[1]
    coef0 = result.params[0]

    res = res.append(pd.DataFrame([[len(y), i, coef0, coef, std, p]], columns=['num_data','snp_idx', 'coef0','coef','std', 'p_value']), ignore_index=True, sort=True)

    # res = res.append(pd.DataFrame([[len(y), i, coef, std, p]], columns=['num_data', 'snp_idx', 'coef', 'std', 'p_value']), ignore_index=True, sort=True)

    return res