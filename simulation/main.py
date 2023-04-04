import numpy as np
import argparse
from simulation import simulation


def main(seeds):
    starts = list(range(0, 10000, 100))
    for i, seed in enumerate(seeds):
        snp_seeds=list(np.arange(starts[i],starts[i]+100))
        for j, snp_seed in enumerate(snp_seeds):
            if j % 10 ==0:
                print("{}th simulation {}th snps.....".format(i, j))
            simulation(config=config, coef=coef, i=i, seed=seed, snp_seed=snp_seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #dgx
    parser.add_argument('--coef_path', type=str, default='../logistic_coef.npy')
    parser.add_argument('--save_path', type=str, default='./')

    parser.add_argument('--N', type=int, default=50000)
    parser.add_argument('--n', type=int, default=5000)
    parser.add_argument('--n_mci', type=int, default=5000)

    parser.add_argument('--maf', type=float, default=0.1)
    parser.add_argument('--snp2mri_ratio', type=int, default=3)

    parser.add_argument('--h_i', type=float, default=0.001)
    parser.add_argument('--kappa', type=float, default=0.1)
    parser.add_argument('--nu', type=float, default=0.9)
    parser.add_argument('--K', type=int, default=55)

    config = parser.parse_args()

    coef = np.load(config.coef_path)
    seeds= list(np.arange(0,100))

    main(seeds)
