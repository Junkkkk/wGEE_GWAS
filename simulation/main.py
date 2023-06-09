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
            simulation(config=config, i=i, seed=seed, snp_seed=snp_seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #dgx
    parser.add_argument('--save_path', type=str, default='./')

    parser.add_argument('--N', type=int, default=50000)
    parser.add_argument('--n_case', type=int, default=5000)
    parser.add_argument('--n_control', type=int, default=5000)
    parser.add_argument('--n_missing', type=int, default=5000)
    parser.add_argument('--n_predictor', type=int, default=50)

    parser.add_argument('--maf', type=float, default=0.1)

    parser.add_argument('--h_i', type=float, default=0.001)
    parser.add_argument('--h_d', type=float, default=0.0001)
    parser.add_argument('--nu', type=float, default=0.5)
    parser.add_argument('--k', type=int, default=10)

    config = parser.parse_args()
    seeds= list(np.arange(0,100))

    main(seeds)
