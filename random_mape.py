import numpy as np
from sklearn.metrics import mean_absolute_percentage_error as mape
import pandas as pd
import time


def extract_non_zeros(test_distr0):

    # set to zero negatives using numpy mask
    test_distr0 = np.array(test_distr0)
    test_distr0 = test_distr0*1*(test_distr0 > 0)
    
    # extract non-zeros
    idx = np.where(1*(test_distr0 > 0) == 1)[0]

    # output only non-zeros
    test_distr = test_distr0[idx]

    # return non-zeros array with idx mapping
    return test_distr, idx


def multi_random_est(test_distr, mape_want=0.2, batch=0.25, thresh=2):

    '''
    mape_want   -   desired mape value
    batch       -   hyperparam for implement batch estimation (use part of an array for estimation)
    thresh      -   stop time for bad estimation
    '''

    # flag for bad estimation
    bad_flag = False
    # temporary copy
    test_distr__ = test_distr.copy()

    # start from test_distr
    new_distr = np.array([test_distr__.copy()]*22)

    # multi-scale, two batch variants
    scales = np.array([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]*2).reshape(-1, 1)
    batches = [0.25]*11 + [0.5]*11

    # first mapes will be 0.0
    mape_check = np.array([mape(test_distr__, nd) for nd in new_distr])
    
    # count time estimation
    start = time.time()

    while list(np.where(np.around(mape_check, 2) == mape_want)[0]) == []:

        # convert to float for implementing random steps
        new_distr = new_distr.astype(np.float64)
        
        # calculate shift including optimization and step reducing
        shift = np.random.uniform(0, 1, size=new_distr.shape) * np.abs(mape_want - mape_check.reshape(-1, 1))**1.1
        shift *= new_distr * np.random.randint(-1, 2, size=new_distr.shape)
        shift /= scales
        
        # implement batch estimation
        mask_zzz = np.array([np.concatenate([np.ones(int(b*len(test_distr))), np.zeros(len(test_distr) - int(b*len(test_distr)))]) for b in batches])
        _ = [np.random.shuffle(z) for z in mask_zzz]
        shift *= mask_zzz

        # two steps because we want to save total sum constant
        new_distr += shift
        _ = [np.random.shuffle(s) for s in shift]
        new_distr -= shift
        

        # avoid negatives
        new_distr = np.abs(new_distr)

        # additional normalizing to speed up convergence
        new_distr = (new_distr/new_distr.sum(axis=1).reshape(-1, 1)*test_distr.sum())
        
        # temprorary thing that helps converge :/
        #new_distr *= 1*(test_distr != 0)
        
        # convert to int for final correction
        new_distr = new_distr.astype(np.int64)
        
        # calculate sum difference
        diff = np.array([test_distr.sum() - new_distr[i].sum() for i in range(new_distr.shape[0])])

        for i, d in enumerate(diff):
            if d <= len(test_distr):
                # random ones shifts
                correct_sum = np.concatenate([np.ones(abs(d)), np.zeros(len(test_distr) - abs(d))])
                np.random.shuffle(correct_sum)
                new_distr[i] += correct_sum.astype(np.int64)

            else:
            # random shifts
                total = d
                temp = []
                for _ in range(len(test_distr)):
                    val = np.random.randint(0, total//2)
                    temp.append(val)
                    total -= val

                correct_sum = np.array(temp)            
                np.random.shuffle(correct_sum)
                new_distr[i] += correct_sum.astype(np.int64)

        # calculate new mape score
        mape_check = np.array([mape(test_distr__, nd) for nd in new_distr])
        
        if time.time() - start > thresh:
            bad_flag = True
            break

    #print(f'MAPE: {mape(test_distr__, new_distr)}, diff: {test_distr.sum() - new_distr.sum()}, exec time: {(time.time() - start)*1000:.2f} ms')
    return new_distr[np.where(np.around(mape_check, 2) == mape_want)], mape_check[np.where(np.around(mape_check, 2) == mape_want)], bad_flag


def multi_get_output(mass, mape_want=0.2):
    # get non-zeros
    test_distr, idx = extract_non_zeros(mass)

    # get output from random estimation
    new_distrs, mapes, flag = multi_random_est(test_distr, mape_want=mape_want)

    # correct output massive using idx from non-zeros extraction
    output = []
    if not flag:
        for k in range(new_distrs.shape[0]):
            new_distr2 = np.zeros(len(mass))

            for i, v in zip(idx, new_distrs[k]):
                new_distr2[i] = v
            
            output.append(new_distr2.astype(np.int64))
        
        return np.array(output), flag