def reverse_some_gt_values(data):
    """Reverse order of two groundtruth questions"""
    """Reads DataFrame"""
    """Returns modified Dataframe"""
    rev_cols = ['qn412','qn416']

    rev_dict1 = {1.0:104, 2.0:103, 3.0:102, 4.0:101}
    rev_dict2 = {104:4, 103:3, 102:2, 101:1}

    for col_name in rev_cols:  #reverse order, perform in 2 batches to prevent overwriting
        for val in [1,2,3,4]:
            data[col_name].replace(val, rev_dict1[val], inplace=True)

    for col_name in rev_cols:  #reverse order, perform in 2 batches to prevent overwriting
        for val in [101,102,103,104]:
            data[col_name].replace(val, rev_dict2[val], inplace=True)
    return data


def sum_raw_gt_values(data):
    """Sum groundtruth questions"""
    """Reads DataFrame"""
    """Returns list"""

    gt_list = []
    gt_binary_list = []

    gt = 0
    for idx, row in data.iterrows():
        gt = 0
        gt += row['qn406']
        gt += row['qn418']
        gt += row['qn407']
        gt += row['qn411']
        gt += row['qn414']
        gt += row['qn420']
        gt += row['qn412']
        gt += row['qn416']
        gt_list.append(gt)

        if gt >= 17:  # 17 out of 32 (about 20%)
            gt_binary_list.append(1)
        else:
            gt_binary_list.append(0)

    return gt_list, gt_binary_list



def map_cfps_dsm(data):  # reads, but doesn't overwrite
    """Map Groundtruth Questions to DSMV"""
    """Reads Dataframe (with Answers to Two Questions Reversed - 'qn412','qn416')"""
    """Returns List with DSM Mapping (see below for DSM Diagnosis Categories)"""

    n, n_col = data.shape
    print(data.shape)
    dsm_diagnosis = []  # list
    dsm_binary = []  # list

    for idx, row in data.iterrows():# for every datapoint - Skip first line (names)
        num_symptoms = 0
        basic_criterion = 0

        # -- Minimum Must-Meet Criterion for Major Depression -------
        # 'qn406': "I'm in a low spirit (Dysphoria)"
        # 'qn418': "I feel sad (Dysphoria & Anhedonia)"
        if row['qn406'] == 4 or row['qn418'] == 4:
            basic_criterion = 1

        # Additional Symptoms for 'qn406' & 'qn418'
        if row['qn406'] == 4 and row['qn418'] == 4:
            num_symptoms += 1

        elif row['qn406'] == 3 and row['qn418'] == 3:
            num_symptoms += 2

        elif row['qn406'] == 3 and row['qn418'] != 3:
            num_symptoms += 1

        elif row['qn406'] != 3 and row['qn418'] == 3:
            num_symptoms += 1

        # Symptoms for 'qn407': "I find it difficult to do anything"
        if row['qn407'] >= 3: num_symptoms += 4

        # Symptoms for 'qn411': "I cannot sleep well"
        if row['qn411'] >= 3: num_symptoms += 2

        # Symptoms for 'qn414': "I feel lonely"
        if row['qn414'] == 4: num_symptoms += 1

        # Symptoms for 'qn420': "I feel that I cannot continue with my life"
        if row['qn420'] >= 3: num_symptoms += 5

        # Symptoms for 'qn412': "I feel happy (REVERSED)"
        if row['qn412'] == 4: num_symptoms += 2

        # Symptoms for 'qn416': "I have a happy life (REVERSED)"
        if row['qn416'] == 4: num_symptoms += 2

        #---------------------------------------------------
        # DSM Diagnosis Categories:
        #  -- 4 --  Major Depressive Episode
        #  -- 3 --  Probably Major Depressive Episode
        #  -- 2 --  Possible Major Depressive Episode
        #  -- 1 --  Subthreshold Depression Symptoms
        #  -- 0 --  No Clinical Significance

        if basic_criterion == 1 and num_symptoms >= 4:
            gt = 2

        elif basic_criterion == 1 and num_symptoms == 3:
            gt = 2

        elif basic_criterion == 1 and num_symptoms == 2:
            gt = 1

        elif basic_criterion != 1 and num_symptoms >= 4:
            gt = 1

        elif basic_criterion != 1 and num_symptoms < 4:
            gt = 0

        else:
            gt = 0

        # append to list
        dsm_diagnosis.append(gt)

    return dsm_diagnosis  #numpy list