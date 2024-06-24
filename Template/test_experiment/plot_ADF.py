#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import os 
from ase.visualize import view
from ase.io import read
from ase import Atoms

def obtainSamples(path, fromExp=False):
    '''
    Obtain the list of samples from the path
    '''
    print('Obtain samples from {}'.format(path)) 
    print('Expriemental data: {}'.format(fromExp))
    if fromExp:
        sampleList = [ 'Ying_Jiang_1',
                'Ying_Jiang_2_1',
                'Ying_Jiang_2_2',
                'Ying_Jiang_3',
                'Ying_Jiang_5',
                'Ying_Jiang_6',
                'Ying_Jiang_4' ] 
        samples = [os.path.join(path, sample + '_mol.xyz') for sample in sampleList]
    else:
        samples =  [os.path.join(path, sample) for sample in os.listdir(path) if sample.endswith('.xyz')]

    print('Number of samples: {}'.format(len(samples)))
    return samples


def read_xyz_with_atomic_numbers(file_path):
    atomic_numbers_to_symbols = {1: 'H', 8: 'O', 29: 'Cu',  79: 'Au'}
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Read number of atoms and comment line
    num_atoms = int(lines[0])
    comment = lines[1].strip()
    #print(comment)

    # Read the atomic data
    symbols = []
    positions = []

    for line in lines[2:2 + num_atoms]:
        parts = line.split()
        atomic_number = int(parts[0])
        x, y, z = map(float, parts[1:4])

        symbol = atomic_numbers_to_symbols[atomic_number]
        symbols.append(symbol)
        positions.append([x, y, z])

    positions = np.array(positions)

    return Atoms(symbols=symbols, positions=positions)



def find_closest_hydrogens(atoms, O_idx,  num_hydrogens=2, mic=True):
    '''
    Find the closest hydrogen atoms to an oxygen atom
    '''
    H_indices = [i for i, atom in enumerate(atoms) if atom.symbol == 'H']
    distances = atoms.get_distances(O_idx, H_indices, mic=mic)
    # Sort based on distance and pick the closest two
    closest_hydrogens = sorted(distances)[:num_hydrogens]
    #print('Closest hydrogens:', closest_hydrogens)

    # Find the indices of the closest hydrogens
    closest_hydrogen_indices = [H_indices[i] for i, d in enumerate(distances) if d in closest_hydrogens]
    return closest_hydrogen_indices

def cal_angles(atoms, A, B, C, mic=True):
    '''
    Calculate angles between atom type A, B, and C, where B is the central atom, A and C are the first and second neighbors
    '''
    symbols = atoms.get_chemical_symbols()
    B_indices = [i for i, s in enumerate(symbols) if s == B]
    angles = []
    for B_idx in B_indices:
        A_idx, C_idx = find_closest_hydrogens(atoms, B_idx, num_hydrogens=2, mic=mic)
        angle = atoms.get_angle(A_idx, B_idx, C_idx, mic=mic)
        angles.append(angle)
    return np.array(angles) 

def adf(ABC_all_angles, dtheta=1):
    '''
    Calculate the angular distribution function for two atom types A and B
    input:
        atoms: ASE Atoms object
        r_max: float, maximum distance to calculate RDF
        dr: float, bin size
    return:
        r: np.array, distance values
        g: np.array, RDF values
    '''
    # Extract positions of atom type A and B
    num_bins = int(180 / dtheta) 
    bins = np.linspace(0, 180, num_bins + 1)

    # Calculate ADF
    ntheta, theta = np.histogram(ABC_all_angles, bins=bins)
    theta = 0.5 * (theta[1:] + theta[:-1]) # r is the center of the bin

    return theta, ntheta


def mean_adf(samples, A, B, C, subNum=79, mic=True):
    '''
    Calculate the mean ADF for a list of samples
    '''
    ABC_all_angles = []
    rhos = []
    refNums = []
    for sample in samples:
        # Prepare the sample atoms with PBC cell 
        atoms = read_xyz_with_atomic_numbers(sample)
        #substrate = atoms[atoms.numbers == subNum]
        #lattice_vectors = calculate_lattice_vectors(substrate)
        #atoms.set_pbc([True, True, True])
        #atoms.set_cell(lattice_vectors)

        # Calculate angles between atom type A, B, and C. 
        angles = cal_angles(atoms, A, B, C, mic=mic)

        # Collect all distances
        ABC_all_angles.append(angles)
    ABC_all_angles = np.concatenate([arr.flatten() for arr in ABC_all_angles])
    theta, ntheta = adf(ABC_all_angles)
    return theta, ntheta 

if __name__ == '__main__':
    path = './predictions_augmentation' if len(os.sys.argv) == 1 else os.sys.argv[1] 
    fromExp = True if len(os.sys.argv) == 1 else False
    samples = obtainSamples(path, fromExp=fromExp)
    theta, adf_HOH = mean_adf(samples, 'H', 'O', 'H', mic=False)
    np.savez('ADF_{}_.npz'.format('predictions'), theta=theta, \
             adf_HOH=adf_HOH)
        
    plt.plot(theta, adf_HOH, label='H-O-H')
    plt.legend()
    plt.xlabel(r'$\theta$ (degree)')
    plt.ylabel('ADF')
    plt.title('ADF of simulation structures for predictions of experiment')
    plt.savefig('ADF_exp.predictions.png')
    #plt.show()


