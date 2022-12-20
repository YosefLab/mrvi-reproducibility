# %%
import argparse


def parser():
    parser = argparse.ArgumentParser(description='Analyze results of symsim_new')
    parser.add_argument('--results_paths', '--list', nargs='+')
    return parser.parse_args()

# %%
args = parser()
results_paths = args.results_paths
for res in results_paths:
    print(res)
