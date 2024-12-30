import gc
import os
from pathlib import Path

import numpy as np
import torch
import random
import yaml

import mlspm.preprocessing as pp
from mlspm import graph, utils
from mlspm.models import GraphImgNetIce
from mlspm.datasets import download_dataset

MM_TO_INCH = 1 / 25.4


def load_data(data_path: os.PathLike):

    # Load data
    data = np.load(data_path)
    X = data["data"][None]
    afm_dim = (data["lengthX"], data["lengthY"])

    # Preprocess
    X = apply_preprocessing([X], afm_dim)

    return X


def make_prediction(model: GraphImgNetIce, X, match_threshold, device):
    with torch.no_grad():
        box_borders = graph.make_box_borders(X.shape[1:3], (model.afm_res, model.afm_res), z_range=model.posnet.grid_z_range)
        xt = torch.from_numpy(X).float().to(device)
        xt, _ = model.posnet(xt.unsqueeze(1))
        pos, matches, labels = graph.find_gaussian_peaks(
            xt, box_borders, match_threshold=match_threshold, std=model.posnet.peak_std, method=model.posnet.match_method
        )
        pred_graph, _ = model.predict_graph(X, pos=pos)
        pred_grid = xt.cpu().numpy()
        matches = matches.cpu().numpy()
        labels = labels.cpu().numpy()
    return pred_graph, pred_grid, matches, labels, box_borders


def apply_preprocessing(X, real_dim):
    X = pp.interpolate_and_crop(X, real_dim)
    pp.add_norm(X)
    X = X[0]
    return X

def rotate_data(X):
    rotations = {
        0: X, 
        90: np.rot90(X, k=1, axes=(1, 2)).copy(), # Copy to avoid negative strides issue
        180: np.rot90(X, k=2, axes=(1, 2)).copy(),
        270: np.rot90(X, k=3, axes=(1, 2)).copy()
    }
    return rotations

if __name__ == "__main__":
    
    graphnet_fit_dir = Path('../train_graphnet')
    posnet_fit_dir = Path('../train_posnet')

    # Get config
    #config_path = Path(graphnet_fit_dir) / 'config.yaml'
    config_path =  'config.yaml'
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    cfg['run_dir'] = './'
    cfg['posnet_weights'] = str(posnet_fit_dir / 'best_model.pth')
    cfg['graphnet_weights'] = str(graphnet_fit_dir / 'best_model.pth')
    cfg['world_size'] = 1

    run_dir = Path(cfg['run_dir'])
    run_dir.mkdir(exist_ok=True, parents=True)
    with open(run_dir / 'config.yaml', 'w') as f:
        # Remember settings
        yaml.safe_dump(cfg, f)

    # Set random seeds
    torch.manual_seed(cfg['random_seed'])
    random.seed(cfg['random_seed'])
    np.random.seed(cfg['random_seed'])

    # # Download the dataset if it's not already there
    # download_dataset(cfg['dataset'], cfg['data_dir'])

    # Start test
    # run(cfg)

    ##############################
    # Prepare Experimatal Images
    ##############################
    exp_data_dir = Path("./exp_data")
    match_thresholds = {
        "cu111": 0.5,
        "au111-monolayer": 0.5,
        "au111-bilayer": 0.6,
        "augmentation": 0.5, # Testing
    }
    classes = [[1], [8]]
    device = "cuda"

    exp_data_files = [
        "Chen_CO.npz",
        "Ying_Jiang_1.npz",
        "Ying_Jiang_2_1.npz",
        "Ying_Jiang_2_2.npz",
        "Ying_Jiang_3.npz",
        "Ying_Jiang_4.npz", # Takes a lot of video memory
        "Ying_Jiang_5.npz",
        "Ying_Jiang_6.npz",
        "Ying_Jiang_7.npz",
    ]

    # Download experimental dataset
    download_dataset("AFM-ice-exp", exp_data_dir)

    #############################
    # Load weight and test 
    #############################
    for weights in ["augmentation"]:
        if weights == "augmentation":
            #
            print('Debug:')
            model = GraphImgNetIce(grid_z_range=(-2.9, 0.5), device=device) 
            model.load_state_dict(torch.load(cfg['graphnet_weights']), strict=False)
            model.posnet.load_state_dict(torch.load(cfg['posnet_weights']))
        else:
            model = GraphImgNetIce(pretrained_weights=weights, device=device)

        # Create out dir
        out_dir = Path(f"predictions")
        out_dir.mkdir(exist_ok=True)

        print(f"Model: {weights}")

        for exp_data_file in exp_data_files:

            save_name = Path(exp_data_file).stem
            print(f"Experiment: {save_name}")

            # Load data
            X_original = load_data(exp_data_dir / exp_data_file)

            # Rotate data and run predictions for each rotation
            rotations = rotate_data(X_original)
            for angle, X in rotations.items(): 
                print(f"Angle: {angle}")
                pred_graph, pred_grid, matches, labels, box_borders = make_prediction(model, X, match_thresholds[weights], device=device)

                try:
                    # Construct xyz array from the graph
                    xyzs = np.concatenate(
                        [
                            pred_graph[0].array(xyz=True),
                            # Take the elements from the first entry of the element list for the predicted class
                            np.array([classes[ind][0] for ind in pred_graph[0].array(class_index=True)[:, 0]])[:, None],
                        ],
                        axis=1,
                    )

                    # Save atom positions
                    utils.write_to_xyz(xyzs, outfile=out_dir / f"{save_name}_d{angle}_mol.xyz", verbose=0)

                    # Save bond information
                    with open(out_dir / f"{save_name}_d{angle}_bonds.txt", "w") as f:
                        for b in pred_graph[0].bonds:
                            f.write(f"{b[0]} {b[1]}\n")
                except:
                    print('No atoms found')

        # Minimize memory usage
        del model
        gc.collect()
        torch.cuda.empty_cache()
