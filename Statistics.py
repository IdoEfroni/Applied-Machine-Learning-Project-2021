import pandas as pd
import scipy
import scipy.stats
import os
from statistics import mean, variance
# pip install scikit-posthocs
import scikit_posthocs as sp
import openpyxl
from openpyxl import *
from openpyxl.styles import NamedStyle
from openpyxl import load_workbook
import math
import numpy as np
import collections
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def create_results_excel():
    """
    creates an excel from all of the csv's
    :return:
    """
    vgg19 = read_csv("VGG19")
    improved = read_csv("Improved")
    mask = read_csv("Mask")
    list_sheets = []
    dataset_names = []
    for df_mask, df_improve, df_vgg in zip(mask, improved, vgg19):
        dataset_names.append(list(df_mask['Dataset Name'])[0])
        df_mask['Algorithm Name'] = 'Masksembles'
        df_vgg['Algorithm Name'] = 'VGG 19'
        df_vgg['Hyper-Parameters Values'] = ""
        df_improve['Algorithm Name'] = 'Transfer Learning'
        df_mask.replace('***', 0)
        df_improve.replace('***', 0)
        df_vgg.replace('***', 0)
        result = pd.concat([df_mask, df_improve, df_vgg])
        list_sheets.append(result)
    writer = pd.ExcelWriter('./Final Results.xlsx', engine = 'openpyxl')
    for df, name in zip(list_sheets, dataset_names):
        df.to_excel(writer, name, index=False)
    writer.save()
    writer.close()


def read_csv(algo_name):
    """
    the function read all of the csv in a directory
    :param algo_name:
    :return:
    """
    list_csv = []
    directory = os.path.join("./Results/", algo_name)
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                df = pd.read_csv(directory+"/"+file)
                list_csv.append(df)
    return list_csv


def calculate_avg_of_cv(df, category):
    """
    the function takes a specific list in a csv, replaces all ***
    with 0 and then using average on the 10 CV in order to generate one result only
    :param df: the df of the dataset
    :param category: the category we want to check like accuracy
    :return: a result of the average
    """
    list_values = list(df[category])
    list_results = [0 if x == "***" or "" else float(x) for x in list_values]
    return mean(list_results)


def aggregate_list_for_algo(list_df, category):
    """
    the function gets a list of df and calculate for each one of them seperatly
    the average of the category and adds it to a final list
    :param list_df: a list of dataframe for each dataset
    :param category: the category we are checking like accuracy
    :return: a final list of average values
    """
    final_values_datasets = []
    for df in list_df:
        value = calculate_avg_of_cv(df, category)
        final_values_datasets.append(value)
    return final_values_datasets


def calculate_friedman(category):
    """
    the function calculates if all of the algorithms are the same
    :param category: the category we are measuring
    :return: prints the statistics and the p-value
    """
    vgg19 = read_csv("VGG19")
    improved = read_csv("Improved")
    mask = read_csv("Mask")
    vgg_l = aggregate_list_for_algo(vgg19, category)
    improved_l = aggregate_list_for_algo(improved, category)
    mask_l = aggregate_list_for_algo(mask, category)
    print()
    print(scipy.stats.friedmanchisquare(vgg_l, improved_l, mask_l))


def find_variance_parameters(categories):
    vgg19 = read_csv("VGG19")
    improved = read_csv("Improved")
    mask = read_csv("Mask")
    dict_params = {}
    for category in categories:
        vgg_l = aggregate_list_for_algo(vgg19, category)
        improved_l = aggregate_list_for_algo(improved, category)
        mask_l = aggregate_list_for_algo(mask, category)
        total_list = vgg_l + improved_l + mask_l
        total_list_l = [0 if math.isnan(x) else x for x in total_list]
        dict_params[category] = variance(total_list_l)
    for key in dict_params:
        print(f"Category: {key}  Variance: {dict_params[key]}")


def calculate_avg_rank_max(vgg_l, improve_l, mask_l):
    """
    checks the average rank in measurements that checks the max value
    :param vgg_l: values of vgg
    :param improve_l: values of transfer learning
    :param mask_l: values of masksembles
    :return: the average rank
    """
    rank_vgg = 0
    rank_improve = 0
    rank_mask = 0
    for v, i, m in zip(vgg_l, improve_l, mask_l):
        temp = [v, i, m]
        sorted(temp)
        rank_mask += temp.index(m)
        rank_improve += temp.index(i)
        rank_vgg += temp.index(v)
    print(f"Average Rank of VGG: {np.round(rank_vgg/len(vgg_l), 4)}"+"\n"+
          f"Average Rank of Masksembles: {np.round(rank_mask/len(mask_l), 4)}"+"\n"+
          f"Average Rank of Transfer Learining:{np.round(rank_improve/len(improve_l), 4)}")




def calculate_post_hoc(category):
    """
    the function calculates post hoc test to decide which algorithm is better
    :param category: the category we are measuring
    :return: a summary of the results
    """
    vgg19 = read_csv("VGG19")
    improved = read_csv("Improved")
    mask = read_csv("Mask")
    vgg_l = aggregate_list_for_algo(vgg19, category)
    improved_l = aggregate_list_for_algo(improved, category)
    mask_l = aggregate_list_for_algo(mask, category)
    all_cat = mask_l + improved_l + vgg_l
    total_list_l = [0 if math.isnan(x) else x for x in all_cat]
    df = pd.DataFrame(total_list_l, columns=[category])

    mask = ['mask' for i in range(0, 20)]
    imp = ['improved' for i in range(0, 20)]
    vgg_n = ['vgg' for i in range(0, 20)]
    all_name = mask + imp + vgg_n
    df['Algorithm'] = all_name
    print(sp.posthoc_ttest(df, val_col=category, group_col='Algorithm', p_adjust='holm'))
    print()
    calculate_avg_rank_max(vgg_l, improved_l, mask_l)


def find_the_best_algo(categories, file_name, highValue=False, report=False):
    """
    a function we use in the report in order to get basic conclusions
    :param category: the category we are measuring
    :return: a table
    """
    vgg19 = read_csv("VGG19")
    improved = read_csv("Improved")
    mask = read_csv("Mask")
    list_df_cat = []
    for category in categories:
        list_dataset_name = []
        vgg_l = aggregate_list_for_algo(vgg19, category)
        improved_l = aggregate_list_for_algo(improved, category)
        mask_l = aggregate_list_for_algo(mask, category)
        list_results = []
        list_best_alg = []
        for v_mask, v_improved, v_vgg, df in zip(mask_l, improved_l, vgg_l, mask):
            name = list(df['Dataset Name'])[0]
            list_dataset_name.append(name)
            if not highValue:
                val = max(v_mask, v_improved, v_vgg)
            else:
                val = min(v_mask, v_improved, v_vgg)
            list_results.append(np.round(val, 3))
            if val == v_mask:
                list_best_alg.append("Masksembles")
            elif val == v_improved:
                list_best_alg.append("Transfer Learning")
            elif val == v_vgg:
                list_best_alg.append("VGG")
        df = pd.DataFrame(list_dataset_name, columns=['Dataset Name'])
        df['Best Algorithm'] = list_best_alg
        df[category] = list_results
        list_df_cat.append(df)
    temp_col_names = list_df_cat[0]['Dataset Name']
    final_df = pd.DataFrame(temp_col_names, columns=['Dataset Name'])
    for df, category in zip(list_df_cat, categories):
        list_final_results = []
        if not report:
            for best_value, algo in zip(list(df[category]), list(df['Best Algorithm'])):
                final_str = algo + ": " + str(best_value)
                list_final_results.append(final_str)
            final_df[category] = list_final_results
        else:
            for best_value in list(df[category]):
                list_final_results.append(best_value)
            final_df[category] = list_final_results
    final_df.to_csv(file_name)
    if report:
        return final_df


def plot_dict(D, title):
    """
    the function get a dictionary and plots bar plot
    :param D: dictionary
    :return: plots
    """
    plt.bar(range(len(D)), list(D.values()), align='center')
    plt.xticks(range(len(D)), list(D.keys()))
    plt.title(title)
    plt.show()

def iterate_hyper_params(list_df, title):
    """
    the function parses the string of the hyper parameters
    :param list_df: a list of df for each dataset
    :return: plots
    """
    list_lr = []
    list_optimizer = []
    list_activation = []
    for df in list_df:
        for cell in list(df['Hyper-Parameters Values']):
            values = cell.split('\n')
            for value in values:
                if 'Activation' in value:
                    act = value.split('Activation: ')
                    list_activation.append(act[1])
                elif 'Learning Rate' in value:
                    lr = value.split('Learning Rate: ')
                    list_lr.append(float(lr[1]))
                elif 'Optimizer' in value:
                    opt = value.split('Optimizer: ')
                    list_optimizer.append(opt[1])
    d_opt = dict(collections.Counter(list_optimizer))
    d_act = dict(collections.Counter(list_activation))
    print(d_opt)
    print(d_act)
    plot_dict(d_opt, 'Optimizer in ' + title)
    plot_dict(d_act,  'Activation in ' + title)


def find_common_hyper_params():
    """
    the function check what are the most common hyper parameters
    :return: 2 plots
    """
    improved = read_csv("Improved")
    mask = read_csv("Mask")
    iterate_hyper_params(improved, 'Transfer Learning')
    iterate_hyper_params(mask, 'Masksembles')


def correlation_between_size_and_cat(category):
    df = find_the_best_algo([category], 'Report_Corr_temp.csv', False, True)
    list_sizes_ds = [380, 304, 228, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 734, 660, 528, 528, 396, 789, 670]
    class_num_ds = [5, 4, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 3, 6, 10]
    corr_size_ds, _ = pearsonr(list(df[category]), list_sizes_ds)
    corr_size_cl, _ = pearsonr(list(df[category]), class_num_ds)
    print(f'Pearsons correlation Number of Classes and {category}: %.3f' % corr_size_cl)
    print(f'Pearsons correlation Number of Records and {category}: %.3f' % corr_size_ds)
    return

# Remove from comment if you want to calculate the Tests
# find_variance_parameters(['AUC', 'Accuracy', 'PR-Curve'])
# print("Friedman Test:")
# calculate_friedman("AUC")
# print()
# print("Post Hoc Test:")
# calculate_post_hoc("AUC")


## Remove from comment if you want to create an excel

# create_results_excel()

# Remove the comment if you want to generate data about some measurement categories

# find_the_best_algo(['Accuracy', 'TPR', 'FPR', 'Precision', 'AUC', 'PR-Curve'], 'Report_Max_Values.csv', False)
# find_the_best_algo(['FPR', 'Training Time', 'Inference Time'], 'Report_Min_Values.csv', True)

# Remove the comment in order to see what are the most common hyper parameters
# find_common_hyper_params()

# Remove comments if you want to calculate the Correlations

# categories = ['Accuracy', 'TPR', 'FPR', 'Precision', 'AUC', 'PR-Curve', 'FPR', 'Training Time', 'Inference Time']
# for category in categories:
    # correlation_between_size_and_cat(category)

