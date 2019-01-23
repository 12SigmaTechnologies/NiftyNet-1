import os
import nibabel as nib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import re


def dice_score(gt, img):
    """Calculate the dice score for evaluation purposes"""
    gt, img = [x > 0 for x in (gt, img)]
    num = 2 * np.sum(gt & img)
    den = gt.sum() + img.sum()
    return num / den


def results(ground_truths, est_dirs, search_pattern='.*bin_seg_(.*_\d+)'):
    """Collates the dice scores from various experiments"""
    result = {e: [] for e in est_dirs}
    result['ids'] = []
    for f in ground_truths:
        r = re.search(search_pattern, f)
        if r:
            gt = imread(f)
            subj_id = r.group(1)
            result['ids'].append(subj_id)
            for exp_name in est_dirs:
                est_path = os.path.join(est_dirs[exp_name], subj_id + '_niftynet_out.nii.gz')
                est = nib.load(est_path).get_data().squeeze()
                result[exp_name].append(dice_score(gt, est))

    df = pd.DataFrame(result)
    return df


def results_long(df, csv_name):
    """Labels the results as from train or validation datasets"""
    d_split = pd.read_csv(csv_name)
    d_split.columns = ('ids', 'fold')
    merged_df = pd.merge(df, d_split)
    df_long = pd.melt(merged_df, id_vars=['ids', 'fold'])
    return df_long


def add_experiment_info_to_datasets(df, est_dirs):
    """adds the experimental information from the training settings"""
    experiment_numbers, flipping, dataset_splits, deforming = [], [], [], []

    for est_dir_key in est_dirs:
        # getting the dataset_split file from the settings_train txt file:
        train_settings = ' '.join([l.strip() for l in open(est_dirs[est_dir_key] + '../settings_training.txt', 'r')])

        experiment_numbers.append(est_dir_key)

        r = re.search('dataset_split_file:\s.*(\d).csv', train_settings)
        dataset_splits.append(r.group(1))

        r = re.search('flipping_axes:\s\((.*?)\)', train_settings)
        flip = 'False' if '-1' in r.group(1) else 'True'
        flipping.append(flip)

        r = re.search('elastic_deformation:\s(\w+)', train_settings)
        deforming.append(r.group(1))

    data_dict = {'variable': experiment_numbers,
                 'flip': flipping,
                 'deform': deforming,
                 'train_split': dataset_splits,
                 'augmentations': ['_'.join(['flip', x[0], 'def', y[0]]) for x, y in zip(flipping, deforming)]
                 }

    conditions_df = pd.DataFrame(data_dict)
    combined_df = pd.merge(df, conditions_df, left_index=True, right_index=True)
    return combined_df


def get_and_plot_results(ground_truths, est_dirs, subj_ids, raw_images=None):
    df = None
    for est_dir_key in est_dirs:

        # getting the dataset_split file from the settings_train txt file:
        train_settings = [l.strip() for l in open(est_dirs[est_dir_key] + '../settings_training.txt', 'r')]
        dataset_split_file = [x.split(':')[1].strip() for x in train_settings if 'dataset_split' in x][0]

        search_pattern = '([^/]*)_binary_mask'
        new_df = results(ground_truths, {est_dir_key: est_dirs[est_dir_key]}, search_pattern=search_pattern)
        new_df_long = results_long(new_df, dataset_split_file)

        f, axes = plt.subplots(len(subj_ids), 1, figsize=(9, 3 * len(subj_ids)))
        if len(subj_ids) == 1:
            axes = [axes]
        f.suptitle("Experiment %s" % est_dir_key)
        show_model_outputs(ground_truths, new_df_long, {est_dir_key: est_dirs[est_dir_key]}, subj_ids, axes, raw_images)

        if df is None:
            df = new_df_long
        else:
            df = pd.concat([df, new_df_long])

    combined_df = add_experiment_info_to_datasets(df, est_dirs)
    return combined_df


def show_model_outputs(ground_truths, df, est_dirs, subj_ids, axes, raw_images=None):
    """Plots the results for visualisation"""
    for est_dir in est_dirs.values():
        for i, sid in enumerate(subj_ids):
            a = imread([f for f in ground_truths if sid in f][0])
            b = nib.load(est_dir + '/' + sid + '_niftynet_out.nii.gz').get_data().squeeze()

            if raw_images is not None:
                # show raw images as well
                c = imread([f for f in raw_images if sid in f][0]) / 255
                axes[i].imshow(np.hstack([c, a, b, a - b]), cmap='gray')
            else:
                axes[i].imshow(np.hstack([a, b, a - b]), cmap='gray')
            axes[i].set_axis_off()

            train_or_val = df[df['ids'] == sid]['fold'].values[0]
            axes[i].set_title('{} Fold: Ground truth, Estimate and Difference. Dice Score = {:.2f}'.format(
                train_or_val, dice_score(a, b)))


def save_visualization(output_dir, ground_truths, est_dirs, subj_ids, raw_images=None):
    """Save hstacked visualization to output_dir"""
    os.makedirs(output_dir, exist_ok=True)
    for sid in subj_ids:
        # save visualization of individual subj_id to file
        plt.clf()
        get_and_plot_results(ground_truths, est_dirs, [sid], raw_images)
        fig = plt.gcf()
        fig.savefig(os.path.join(output_dir, sid + '_vis.png'))

