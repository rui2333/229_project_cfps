import numpy as np
import time
import os

data_path = data_path =r'/Users/~/data_dir'
code_path = r'/Users/~/code_dir'

os.chdir(code_path)

import softmax_reg
import PerformMetrics1

os.chdir(data_path)

print('data_path', data_path)

x_all = np.loadtxt("train.csv", skiprows=1, delimiter=",")
y_all = np.loadtxt("dsm_diagnosis_3b_cats.csv", delimiter=",")

(n, n_col) = x_all.shape

# Set Type of Analysis:

analysis = 'forward_search'
# analysis = 'backward_search'
analysis = 'ablation_analysis'
analysis = 'baseline'


if analysis in ['forward_search', 'backward_search', 'ablation_analysis']:
    Weighted_PPV = np.zeros(n_col)
    PPV_by_cats = np.zeros((n_col,3))  # 3 categories
    accuracy_by_cats = np.zeros((n_col,3))  # 3 categories

elif analysis == 'baseline':
    Weighted_PPV_0 = np.zeros(1)  # Baseline
    PPV_by_cats_0 = np.zeros((1,3))  # 3 categories
    accuracy_by_cats_0 = np.zeros((1,3))  # 3 categories

#-------------------------------
n_cols_list = np.arange(n_col)
row_indices = list(np.arange(n))

np.random.seed(229)
np.random.shuffle(row_indices)  # shuffle twice
np.random.shuffle(row_indices)

n_test = int(0.95 * n)  # reserve a test set

tranche_test = row_indices[n_test: ]
tranche_1 = row_indices[: int(0.20 * n_test)]
tranche_2 = row_indices[int(0.20 * n_test): int(0.40 * n_test)]
tranche_3 = row_indices[int(0.40 * n_test): int(0.60 * n_test)]
tranche_4 = row_indices[int(0.60 * n_test): int(0.80 * n_test)]
tranche_5 = row_indices[int(0.80 * n_test): n_test]

#--Not Iterated--
train_1 = tranche_1 + tranche_3 + tranche_4 + tranche_5
val_1 = tranche_2

train_2 = tranche_1 + tranche_2 + tranche_4 + tranche_5
val_2 = tranche_3

train_3 = tranche_1 + tranche_2 + tranche_3 + tranche_5
val_3 = tranche_4

train_4 = tranche_1 + tranche_2 + tranche_3 + tranche_4
val_4 = tranche_5

train_5 =  tranche_2 + tranche_3 + tranche_4 + tranche_5
val_5 = tranche_1

train_set = [train_1,train_2,train_3, train_4, train_5]
val_set = [val_1,val_2,val_3,val_4,val_5]


#
if analysis == 'baseline':
    iter_outer_loop = range(1)

elif analysis == 'forward_search':
    iter_outer_loop = range(n_col)

elif analysis == 'backward_search':
    iter_outer_loop = range(1,n_col)

elif analysis == 'ablation_analysis':
    iter_outer_loop = range(n_col)

# Start k-Fold Cross Validation
for icol in iter_outer_loop:
    if analysis == 'baseline':
        new_col_indices = n_cols_list

    elif analysis == 'forward_search': # FS: Columns To Add (Cumulatively), Ending at icol # FS: Add Columns starting at 1
        new_col_indices = range(icol+1)

    elif analysis == 'backward_search':  # BS: Columns to Drop (Cumulatively) # BS: Drop Columns starting at 0
         new_col_indices = range(icol,n_col)

    elif analysis == 'ablation_analysis': # ABLATION: Drop One Column at a Time
        new_col_indices = n_cols_list
        new_col_indices = np.delete(n_cols_list, icol)

    x_all_MOD = x_all[:,new_col_indices]
    print(x_all_MOD.shape)

    y_train_total = []
    y_val_total = []
    y_pred_train_total = []
    y_pred_val_total = []

    k_fold = 0
    for train_tranche, val_tranche in zip(train_set, val_set):

        x_train = x_all_MOD[train_tranche,:] # All Factors Only - 175 to where values get very small
        y_train = y_all[train_tranche]

        x_val = x_all_MOD[val_tranche,:]
        y_val = y_all[val_tranche]

        k_fold += 1

        # -----SOFTMAX REGRESSION----
        y_pred_train, y_pred_val = softmax_reg.run_softmax_reg(x_train, y_train, x_val, y_val)

        y_train_total += list(y_train)  # Add to List from np.array
        y_val_total += list(y_val)
        y_pred_train_total += list(y_pred_train)
        y_pred_val_total += list(y_pred_val)

    # After Convergence of All k-Folds
    print("After Convergence:", icol)

    print('Train:', PerformMetrics1.run_multiclass_metric(y_pred_train_total,y_train_total))
    print('Validation:', PerformMetrics1.run_multiclass_metric(y_pred_val_total,y_val_total))

    # RETURN: PREDICT ONLY: scalar, np.array, np.array

    if analysis == 'baseline':
        Weighted_PPV_0[0], PPV_by_cats_0[0] = PerformMetrics1.run_multiclass_metric(y_pred_val_total,y_val_total)

        np.save('Weighted_PPV_0',Weighted_PPV_0)  # file name, array
        np.save('PPV_by_cats_0',PPV_by_cats_0)


    elif analysis == 'forward_search': # FS: Columns To Add (Cumulatively), Ending at icol # FS: Add Columns starting at 1
        Weighted_PPV[icol], PPV_by_cats[icol] = PerformMetrics1.run_multiclass_metric(y_pred_val_total,y_val_total)

    elif analysis == 'backward_search':  # BS: Columns to Drop (Cumulatively) # BS: Drop Columns starting at 0
         Weighted_PPV[icol-1], PPV_by_cats[icol-1] = PerformMetrics1.run_multiclass_metric(y_pred_val_total,y_val_total)

    elif analysis == 'ablation_analysis': # ABLATION: Drop One Column at a Time
        Weighted_PPV[icol], PPV_by_cats[icol] = PerformMetrics1.run_multiclass_metric(y_pred_val_total,y_val_total)


if analysis != 'baseline':
    np.save('Weighted_PPV',Weighted_PPV)
    np.save('PPV_by_cats',PPV_by_cats)






























