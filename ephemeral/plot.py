from os.path import join
from topology import load_persistence_diagram_dipha, load_persistence_diagram_json
from utilities import parse_filename, get_patient_ids_and_times
import numpy as np
from glob import glob
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

import argparse

def plot_pds(data_path: str, output_path: str='../figures/', dimensions: list=[0,1,2], task: str='pixar', masked: bool=True):
    '''
    Plots the persistence diagrams of all available patients
    in one plot. 
    '''

    all_patients = get_patient_ids_and_times(data_path, task)
    for dimension in dimensions:
        fig = plt.figure(figsize=(40, 35))
        min_c = float('inf')
        min_d = float('inf')
        max_c = -float('inf')
        max_d = -float('inf')
        for pat_idx, pat in tqdm(enumerate(list(all_patients.keys()))):
            ax = fig.add_subplot(6, 5, pat_idx+1, projection='3d')
            ax.set_title(f"Subject: {pat}")

            ax.view_init(17, -60)
            patient = {'c': [], 'd': [], 't': [], 'l': []} # Creation, Destruction, Time, Length

            # Iterate in reverse order to have the first time point plotted on top
            for time in all_patients[pat][::-1]:
                f = join(data_path, f'sub-{task}{pat}_task-{task}_bold_space-MNI152NLin2009cAsym_preproc_{time}.json')

                dim, creation, desctruction = load_persistence_diagram_json(f)

                c = creation[np.argwhere(dim == dimension)].ravel()
                d = desctruction[np.argwhere(dim == dimension)].ravel()

                patient['c'] += c.tolist()
                patient['d'] += d.tolist()
                patient['t'] += [float(time)] * len(c)
                patient['l'] += [len(c)]
                
                min_c = np.min([min_c, np.min(c)])
                min_d = np.min([min_d, np.min(d)])
                max_c = np.max([max_c, np.max(c)])
                max_d = np.max([max_d, np.max(d)])
        
            # Generate color palette based on the number of time steps.
            palette = sns.color_palette('Spectral_r', len(np.unique(patient['t'])))
            colors = []

            for i, l in enumerate(patient['l']):
                colors += [palette[i]]*l

            ax.scatter(patient['c'], patient['t'], patient['d'], c=colors, alpha=0.8, s=8)
            ax.set_xlabel('Birth')
            ax.set_ylabel('Time')
            ax.set_zlabel('Death')
            
        for ax in fig.get_axes():
            ax.set_xlim(min_c, max_c)
            ax.set_zlim(min_d, max_d)
        plt.tight_layout()
        plt.savefig(join(output_path, f'{"masked" if masked else "raw"}_dimension_{dimension}.png'))
    fig = plt.figure(figsize=(40, 35))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', help="Path to .json files.")
    parser.add_argument('--raw', action='store_false', help="Whether to plot the raw data.")
    parser.add_argument('-d', '--dimensions', nargs='+', default=[0,1,2], help="Which dimensions to plot.")
    
    args = parser.parse_args()
    plot_pds(data_path=args.INPUT, dimensions=args.dimensions, masked=args.raw)

