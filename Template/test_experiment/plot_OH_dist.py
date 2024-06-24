#!/usr/bin/env python
# %%
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
from ase.visualize import view
from ase.io import read
from ase import Atoms
import os, sys



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

# %%
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


def calculate_lattice_vectors(atoms):
    '''
    Calculate lattice vectors for a system 
    with non-orthogonal XY plane and orthogonal Z
    input: 
        atoms: ASE Atoms object for the substrate
    return: 
        Lattice vectors
    '''
    positions = atoms.positions
    xy_positions = positions[:, :2]
    atomNum = xy_positions.shape[0]
    # print(f'Number of atoms: {atomNum}')

    miny = np.min(xy_positions[:, 1])
    maxy = np.max(xy_positions[:, 1])
    first_row = xy_positions[np.where(xy_positions[:, 1] == miny)]
    last_row = xy_positions[np.where(xy_positions[:, 1] == maxy)]
    #print(f'First row:\n {first_row}', first_row.shape)
    #print(f'Last row:\n {last_row}', last_row.shape)

    if first_row.shape[0] != last_row.shape[0]:
        raise ValueError("The number of atoms in the first row is not equal to the number of atoms in the last row.")

    atomNumFirstRow = first_row.shape[0]
    spaceX = (first_row[-1][0] - first_row[0][0]) / (atomNumFirstRow - 1)
    a = np.array([atomNumFirstRow * spaceX, 0, 0])
    #print('Vector x', a)

    atomNumPerRow = first_row.shape[0]
    atomNumPerCol = atomNum // atomNumPerRow
    #print(f'Number of atoms per column: {atomNumPerCol}')
    spaceY = (last_row[0][1] - first_row[0][1]) / (atomNumPerCol - 1)
    b = np.array([last_row[0][0] + spaceX/2, atomNumPerCol * spaceY, 0])
    #print('Vector y', b)

    # The z vector is orthogonal and fixed
    c = np.array([0, 0, 30.0])
    
    # Combine vectors to form the lattice matrix
    lattice_vectors = np.array([a, b, c])
    return lattice_vectors

def read_samples(simulation, toSee):
    '''
    Read a list of sample paths from a list of simulations
    Input: 
        simulations: a folder
        toSee: tuple of two integers, number of folders and number of xyz files to see  
    Output: a list of sample paths
    '''
    samples = []
    train_folders = [f for f in os.listdir(simulation) if 'train' in f]
    folderToSee = int(toSee[0]) 
    for train_folder in train_folders[0:folderToSee]:
        train_folder_path = os.path.join(simulation, train_folder)
        xyz_files = [f for f in os.listdir(train_folder_path) if f.endswith('.xyz')]
        xyzToSee = int(toSee[1])
        for xyz_file in  xyz_files[0:xyzToSee]:
            structure = os.path.join(train_folder_path, xyz_file) 
            samples.append(structure)
    return samples

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

def cal_distances(atoms, A, B, r_max=4, mic=True, onlyDistances=False):
    '''
    Calculate distances between atom type A and B
    '''
    positions = atoms.positions
    symbols = atoms.get_chemical_symbols()
    A_indices = [i for i, s in enumerate(symbols) if s == A]
    B_indices = [i for i, s in enumerate(symbols) if s == B]
    B_positions = positions[B_indices]

    # Create a 2D array to store all distances between A and B atoms
    sizeShape = (len(A_indices), len(B_indices)) if A != B else (len(A_indices), len(B_indices)-1) 
    AB_all_distance = []
    for count, A_idx in enumerate(A_indices):
        B_indices_ = [i for i in B_indices if i != A_idx] if A == B else B_indices # Exclude the self pair
        distances = atoms.get_distances(A_idx, B_indices_, mic=mic) # mic=True to consider PBC, where Cell and PBC need to be set
        # Only leave the distances within r_max
        distances = [d for d in distances if d <= r_max]
        AB_all_distance.extend(distances)
    
    if onlyDistances:
        # Only return the distances
        return AB_all_distance
    else:
        # Obtain the density rho of B particles 
        z_max = np.max(B_positions[:, 2])
        z_min = np.min(B_positions[:, 2])
        ratio = (z_max - z_min) / atoms.cell[2][2]
        effective_volume = atoms.get_volume() * ratio
        rho = len(B_indices) / effective_volume

        # Reference numbers
        refNum = len(A_indices)

        return AB_all_distance, rho,  refNum 

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

def distance_dis(AB_all_distances, dr=0.1, r_max=4):
    '''
    Calculate the distance distribution for two atom types A and B
    input:
        AB_all_distances: np.array, all distances between atom type A and B
        dr: float, bin size
        r_max: float, maximum distance to consider
    return:
        r: np.array, distance values
        nr: np.array, distance distribution values
    '''
    # Extract positions of atom type A and B
    num_bins = int(r_max / dr) 
    bins = np.linspace(0, r_max, num_bins + 1)

    # Calculate  distance distribution
    nr, r = np.histogram(AB_all_distances, bins=bins, range=(0, r_max))
    r = 0.5 * (r[1:] + r[:-1]) # r is the center of the bin

    return r, nr 

def mean_distance_distribution(samples, A, B, subNum=79, dr=0.1, r_max=4, mic=True, onlyDistances=False):
    '''
    Calculate the mean ADF for a list of samples
    '''
    AB_all_distances = []
    refNums = []
    for sample in samples:
        # Prepare the sample atoms with PBC cell 
        atoms = read_xyz_with_atomic_numbers(sample)
        if mic:
            substrate = atoms[atoms.numbers == subNum]
            lattice_vectors = calculate_lattice_vectors(substrate)
            atoms.set_pbc([True, True, True])
            atoms.set_cell(lattice_vectors)

        # Calculate distances between atom type A and B. 
        distances = cal_distances(atoms, A, B, r_max=r_max, mic=True, onlyDistances=True)

        # Collect all distances
        AB_all_distances.extend(distances)
    if onlyDistances:
        return AB_all_distances
    else:
        AB_all_distances = np.concatenate([arr.flatten() for arr in AB_all_distances])
        r, nr = distance_dis(AB_all_distances, dr=dr, r_max=r_max)
        return r, nr


if __name__ == '__main__':
    path = './predictions_augmentation' if len(os.sys.argv) == 1 else os.sys.argv[1] 
    fromExp = True if len(os.sys.argv) == 1 else False
    samples = obtainSamples(path, fromExp=fromExp)

    r_max = 10
    colors = ['#F99005', '#0FA842']

    distances = mean_distance_distribution(samples, 'O', 'H', mic=False, dr=0.1, r_max=r_max, onlyDistances=True)
    plt.hist(distances, density=True, range=(0, r_max), color = colors[0], bins=50, alpha=0.5, label='O-H distance distribution')
    plt.hist(distances, histtype='step', fill=False,  density=True, range=(0, r_max), color = colors[0], bins=50, alpha=1)
    plt.xlim(0, r_max)
    plt.legend()
    plt.xlabel(r"O-H distance [$\AA$]")
    plt.ylabel('Density')
    plt.savefig('OH_dist_exp.png')
    plt.show()

    distances = mean_distance_distribution(samples, 'O', 'O', mic=False, dr=0.1, r_max=r_max, onlyDistances=True)
    plt.hist(distances, density=True, range=(0, r_max), color = colors[1], bins=50, alpha=0.5, label='O-O distance distribution')
    plt.hist(distances, histtype='step', fill=False,  density=True, range=(0, r_max), color = colors[1], bins=50, alpha=1)
    plt.xlim(0, r_max)
    plt.legend()
    plt.xlabel(r"O-O distance [$\AA$]")
    plt.ylabel('Density')
    plt.savefig('OO_dist_exp.png')
    plt.show()
