import pandas as pd
from tqdm import tqdm
import torch
from omegaconf import OmegaConf

import os
import sys
sys.path.append('../')
from datasets import SiteSaturationDataset
from thermompnn_benchmarking import compute_centrality, get_trained_model


def retrieve_best_mutants(df_slice, allow_cys=True):
    # check for best mutant at each position
    pos_list = df_slice.position.unique()
    best_res_list = []
    for p in pos_list:
        p_slice = df_slice.loc[df_slice['position'] == p].reset_index(drop=True)
        if not allow_cys:  # filter out cysteine option
            p_slice = p_slice.loc[p_slice['mutation'] != 'C'].reset_index(drop=True)
        min_row = p_slice.iloc[pd.to_numeric(p_slice['ddG_pred']).idxmin()]
        best_res_list.append(min_row['mutation'])
    return best_res_list


def main(cfg, args):
    """Inference script that does site-saturation mutagenesis for a given protein"""
    # define config for model loading
    config = {
        'training': {
            'num_workers': 8,
            'learn_rate': 0.001,
            'epochs': 100,
            'lr_schedule': True,
        },
        'model': {
            'hidden_dims': [64, 32],
            'subtract_mut': True,
            'num_final_layers': 2,
            'freeze_weights': True,
            'load_pretrained': True,
            'lightattn': True,
            'lr_schedule': True,
        }
    }

    cfg = OmegaConf.merge(config, cfg)

    models = {
        'ThermoMPNN': get_trained_model(model_name=args.model_path,
                                        config=cfg,
                                        override_custom=True)
    }
    datasets = {}
    chain_id = 'A'
    if args.structure_dir:
        datasets["SSM-PDB"] = SiteSaturationDataset(structure_dir=args.structure_dir, chain_id=chain_id)

    max_batches = None

    for name, model in models.items():
        model = model.eval()
        model = model.cuda()
        for dataset_name, dataset in datasets.items():
            print('Running model %s on dataset %s' % (name, dataset_name))
            for i, batch in enumerate(tqdm(dataset)):
                raw_pred_df = pd.DataFrame(columns=['model', 'dataset', 'pdb', 'chain', 'position', 'wildtype', 'mutation',
                                                    'ddG_pred', 'neighbors'])
                mut_pdb, mutations = batch
                pdb_id = mut_pdb[0].get("name", f"protein_{i}")
                final_mutation_list = mutations

                pred, _ = model(mut_pdb, final_mutation_list)

                # calculation of N neighbors
                if args.centrality:
                    coord_chain = [c for c in mut_pdb[0].keys() if 'coords' in c][0]
                    chain = coord_chain[-1]
                    neighbors = compute_centrality(mut_pdb[0][coord_chain], basis_atom='CA', backup_atom='C', chain=chain, radius=10.)
                
                for mut, out in zip(final_mutation_list, pred):
                    if mut is not None:
                        raw_pred_df.loc[len(raw_pred_df)] = {
                            'ddG_pred': out["ddG"].cpu().item(),
                            'position': mut.position,
                            'wildtype': mut.wildtype,
                            'mutation': mut.mutation,
                            'pdb': mut.pdb.strip('.pdb'),
                            'chain': chain_id,
                            'model': name,
                            'dataset': dataset_name,
                            'neighbors': neighbors[mut.position].cpu().item() if args.centrality else None,
                            'best_AA': ''  # placeholder for later
                        }

                if not args.include_cys:
                    raw_pred_df = raw_pred_df.loc[raw_pred_df['mutation'] != 'C']

                if max_batches is not None and i >= max_batches:
                    break

                print('Completed protein:', mut.pdb)
                print('Mutations processed:', raw_pred_df.shape)

                os.makedirs(args.outdir, exist_ok=True)
                outfile = os.path.join(args.outdir, f"{name}_{pdb_id}_preds.csv")
                if not args.compressed:
                    raw_pred_df.to_csv(outfile)
                else:
                    raw_pred_df.to_csv(outfile + ".gz", index=False, compression="gzip")
                del raw_pred_df
                print(f"Saved predictions to {outfile}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../local.yaml',
                    help='Path to YAML configuration file (default: ../local.yaml)')
    parser.add_argument('--structure_dir', type=str, default=None, 
                        help='Path to directory containing PDB structure files')
    parser.add_argument('--model_path', type=str, default='../models/thermoMPNN_default.pt',
                        help='Path to ThermoMPNN .pt model file')
    parser.add_argument('--outdir', type=str, default='./',
                    help='Output directory where predictions will be saved')
    parser.add_argument('--include_cys', action='store_true', default=False,
                        help='Include cysteine as potential mutation option.'
                             'Due to assay artifacts, mutations to cys are predicted poorly.')
    parser.add_argument('--centrality', action='store_true', default=False,
                        help='Calculate centrality value for each residue (# neighbors). '
                             'Only used if --keep_preds is enabled.')
    parser.add_argument('--compressed', action='store_true', default=False,
                        help='Compress final csv files')
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    with torch.no_grad():
        main(cfg, args)
