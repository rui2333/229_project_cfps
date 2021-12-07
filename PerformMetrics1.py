import os
import numpy as np
import time

def run_multiclass_metric(y_pred, y_target):     
    #------------------------------------------------#
    # Calculate Weighted PPV
    #------------------------------------------------#        
    unique_vals = set(y_target)
    true_cats = list(unique_vals)    
    true_cats.sort()
    
    n = len(y_target) 
    n_col = len(unique_vals) 
        
    PPV_by_cats = np.zeros(n_col)
        
    y_target = np.array(y_target)  # convert back to an array
    y_pred = np.array(y_pred)
                
    for j,icat in enumerate(true_cats):  
    
        subset_target = y_target[y_target == icat]
        subset_pred = y_pred[y_target == icat]
        n_true_cat = len(subset_target)
        n_correct = 0
        
        n_pred_cat = len(y_pred[y_pred == icat])
        
        for i in range(n_true_cat):
            if subset_target[i] == subset_pred[i]:
                n_correct += 1
        
        if n_pred_cat != 0:
            ppv_cat = n_correct/n_pred_cat  
        else:
           ppv_cat = 0.0 #np.nan     
           
        PPV_by_cats[j] = ppv_cat
    
    
    PPV_weights = np.array((0.15,0.35, 0.50))    
    Weighted_PPV = np.dot(PPV_weights, PPV_by_cats)

    return Weighted_PPV, PPV_by_cats  # scalar, np.array

    









 