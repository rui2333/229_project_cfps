import matplotlib.pyplot as plt
import numpy as np

def add_intercept(x):
    """Add intercept to matrix x.

    Args:
        x: 2D NumPy array.

    Returns:
        New matrix same as x with 1's in the 0th column.
    """
    new_x = np.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
    new_x[:, 0] = 1
    new_x[:, 1:] = x

    return new_x


def load_dataset(csv_path, label_col='y', add_intercept=False):
    """Load dataset from a CSV file.

    Args:
         csv_path: Path to CSV file containing dataset.
         label_col: Name of column to use as labels (should be 'y' or 't').
         add_intercept: Add an intercept entry to x-values.

    Returns:
        xs: Numpy array of x-values (inputs).
        ys: Numpy array of y-values (labels).
    """

    def add_intercept_fn(x):
        global add_intercept
        return add_intercept(x)

    # Validate label_col argument
    allowed_label_cols = ('y', 't')
    if label_col not in allowed_label_cols:
        raise ValueError('Invalid label_col: {} (expected {})'
                         .format(label_col, allowed_label_cols))

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    l_cols = [i for i in range(len(headers)) if headers[i] == label_col]
    inputs = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols)
    labels = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=l_cols)

    if inputs.ndim == 1:
        inputs = np.expand_dims(inputs, -1)

    if add_intercept:
        inputs = add_intercept_fn(inputs)

    return inputs, labels


def plot(x, y, theta, save_path, correction=1.0):
    """Plot dataset and fitted logistic regression parameters.

    Args:
        x: Matrix of training examples, one per row.
        y: Vector of labels in {0, 1}.
        theta: Vector of parameters for logistic regression model.
        save_path: Path to save the plot.
        correction: Correction factor to apply, if any.
    """
    # Plot dataset
    plt.figure()
    plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2)
    plt.plot(x[y == 0, -2], x[y == 0, -1], 'go', linewidth=2)

    # Plot decision boundary (found by solving for theta^T x = 0)
    x1 = np.arange(min(x[:, -2]), max(x[:, -2]), 0.01)
    x2 = -(theta[0] / theta[2] + theta[1] / theta[2] * x1
           + np.log((2 - correction) / correction) / theta[2])
    plt.plot(x1, x2, c='red', linewidth=2)
    plt.xlim(x[:, -2].min()-.1, x[:, -2].max()+.1)
    plt.ylim(x[:, -1].min()-.1, x[:, -1].max()+.1)

    # Add labels and save to disk
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig(save_path)

def normalize(x):

    xshp = np.shape(x)
    m = xshp[0]
    d = xshp[1] 
    for i in range(d-1):
        x_mean = np.mean(x[:,i+1])
        x_stdev = np.std(x[:,i+1])
        x[:,i+1] -= x_mean
        if (x_stdev > 0):
            x[:,i+1] /= x_stdev

    return x

def create_reg(x, y, threshold, log_flag):

    # Statistics
    xshp = np.shape(x)
    m = xshp[0]
    d = xshp[1]
    y_mean = np.mean(y)
    y_stdev = np.std(y)
    y_threshold = y_mean + (threshold*y_stdev)
    print('y stats: mean = ',y_mean,' : stdev = ',y_stdev,' : threshold = ',y_threshold)
    yp = np.zeros(np.shape(y))

    # Linear vs. Logistic Regression
    if (log_flag == 1):
        pos_idx = np.where(y >= y_threshold)
        yp[pos_idx[0]] = 1
        neg_idx = np.where(y < y_threshold)
        print('Logistic Labels')
    else:
        yp = y
        pos_idx = np.where(y >= y_threshold)
        neg_idx = np.where(y < y_threshold)
        print('Linear Regression Labels')

    # Check label percentage
    num_pos = len(pos_idx[0])
    num_neg = len(neg_idx[0])
    pos_overall = num_pos / m
    num_neg = m - num_pos
    neg_overall = num_neg / m
    print('Pos label percentage = ',pos_overall)
    print('Neg label percentage = ',neg_overall)

    # Balance
    m = 2*min(num_pos,num_neg)
    ynew = np.zeros((m,1))
    xnew = np.zeros((m,d))
    pidx = pos_idx[0]
    nidx = neg_idx[0]
    if (num_neg > num_pos):
        for i in range(num_pos):
            ynew[i] = yp[pidx[i]]
            xnew[i,:] = x[pidx[i],:]
            ynew[i+num_pos] = yp[nidx[i]]
            xnew[i+num_pos,:] = x[nidx[i],:]
    else:
        for i in range(num_pos):
            ynew[i] = yp[nidx[i]]
            xnew[i,:] = x[nidx[i],:]
            ynew[i+num_neg] = yp[pidx[i]]
            xnew[i+num_neg,:] = x[pidx[i]]

    return xnew, ynew, y_threshold

def create_nn(x, y, threshold):

    # Statistics
    xshp = np.shape(x)
    m = xshp[0]
    d = xshp[1]
    y_mean = np.mean(y)
    y_stdev = np.std(y)
    t = y_mean + (threshold*y_stdev)
    print('y stats: mean = ',y_mean,' : stdev = ',y_stdev,' : t = ',t)
    yp = np.zeros((m,2))

    # Dual state output
    pos_idx = np.where(y >= t)
    neg_idx = np.where(y < t)
    yp[pos_idx[0],1] = 1
    yp[neg_idx[0],0] = 1
    print('Deep Neural Network - Logistic Labels')

    # Check label percentage
    num_pos = len(pos_idx[0])
    num_neg = len(neg_idx[0])
    pos_overall = num_pos / m
    num_neg = m - num_pos
    neg_overall = num_neg / m
    print('Pos label percentage = ',pos_overall)
    print('Neg label percentage = ',neg_overall)

    # Balance
    m = 2*min(num_pos,num_neg)
    ynew = np.zeros((m,2))
    xnew = np.zeros((m,d))
    pidx = pos_idx[0]
    nidx = neg_idx[0]
    if (num_neg > num_pos):
        for i in range(num_pos):
            ynew[i,:] = yp[pidx[i],:]
            xnew[i,:] = x[pidx[i],:]
            ynew[i+num_pos,:] = yp[nidx[i],:]
            xnew[i+num_pos,:] = x[nidx[i],:]
    else:
        for i in range(num_pos):
            ynew[i,:] = yp[nidx[i],:]
            xnew[i,:] = x[nidx[i],:]
            ynew[i+num_neg,:] = yp[pidx[i],:]
            xnew[i+num_neg,:] = x[pidx[i]]

    return xnew, ynew

def create_sets_nn(x,y,m_train,m_valid,m_test):

    xshp = np.shape(x)
    m = xshp[0]
    d = xshp[1]
    yshp = np.shape(y)
    k = yshp[1]

    # Randomize data set
    rnd_idx = np.uint(np.random.permutation(m))

    # Create Training, Validation, Test Sets
    x_train = np.zeros((m_train,d))
    y_train = np.zeros((m_train,k))
    x_valid = np.zeros((m_valid,d))
    y_valid = np.zeros((m_valid,k))
    x_test = np.zeros((m_test,d))
    y_test = np.zeros((m_test,k))
    for i in range(m_train):
        x_train[i,:] = x[rnd_idx[i],:]
        y_train[i,:] = y[rnd_idx[i],:]
    for i in range(m_valid):
        x_valid[i,:] = x[rnd_idx[m_train+i],:]
        y_valid[i,:] = y[rnd_idx[m_train+i],:]
    for i in range(m_test):
        x_test[i,:] = x[rnd_idx[m_train+m_valid+i],:]
        y_test[i,:] = y[rnd_idx[m_train+m_valid+i],:]

    # Check
    pos_overall = np.count_nonzero(y[:,1]==1)
    pos_train = np.count_nonzero(y_train[:,1]==1)
    pos_valid = np.count_nonzero(y_valid[:,1]==1)
    pos_test = np.count_nonzero(y_test[:,1]==1)

    return x_train, y_train, x_valid, y_valid, x_test, y_test

def create_sets(x,y,m_train,m_valid,m_test,y_threshold,log_flag):

    xshp = np.shape(x)
    m = xshp[0]
    d = xshp[1]

    # Randomize data set
    rnd_idx = np.uint(np.random.permutation(m))

    # Create Training, Validation, Test Sets
    x_train = np.zeros((m_train,d))
    y_train = np.zeros((m_train,1))
    x_valid = np.zeros((m_valid,d))
    y_valid = np.zeros((m_valid,1))
    x_test = np.zeros((m_test,d))
    y_test = np.zeros((m_test,1))
    for i in range(m_train):
        x_train[i,:] = x[rnd_idx[i],:]
        y_train[i] = y[rnd_idx[i]]
    for i in range(m_valid):
        x_valid[i,:] = x[rnd_idx[m_train+i],:]
        y_valid[i] = y[rnd_idx[m_train+i]]
    for i in range(m_test):
        x_test[i,:] = x[rnd_idx[m_train+m_valid+i],:]
        y_test[i] = y[rnd_idx[m_train+m_valid+i]]

    # Check
    if (log_flag == 1):
        pos_overall = np.count_nonzero(y==1)
        pos_train = np.count_nonzero(y_train==1)
        pos_valid = np.count_nonzero(y_valid==1)
        pos_test = np.count_nonzero(y_test==1)
    else:
        pos_overall = np.count_nonzero(y >= y_threshold)
        pos_train = np.count_nonzero(y_train >= y_threshold)
        pos_valid = np.count_nonzero(y_valid >= y_threshold)
        pos_test = np.count_nonzero(y_test >= y_threshold)

    return x_train, y_train, x_valid, y_valid, x_test, y_test

def analyze_prediction(pred, y, y_threshold, log_flag):

    m = len(y)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    if (log_flag == 1):
        for k in range(m):
            if ((y[k]==1) & (pred[k]>=0.5)):
                tp += 1
            if ((y[k]==0) & (pred[k]>=0.5)):
                fp += 1
            if ((y[k]==0) & (pred[k]<0.5)):
                tn += 1
            if ((y[k]==1) & (pred[k]<0.5)):
                fn += 1
    else:
        for k in range(m):
            if ((y[k]>y_threshold) & (pred[k]>=y_threshold)):
                tp += 1
            if ((y[k]<y_threshold) & (pred[k]>=y_threshold)):
                fp += 1
            if ((y[k]<y_threshold) & (pred[k]<y_threshold)):
                tn += 1
            if ((y[k]>y_threshold) & (pred[k]<y_threshold)):
                fn += 1
                
    accuracy = (tp+tn)/m
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    fscore = 2 / ((1/precision)+(1/recall))

    return accuracy, precision, recall, fscore

def analyze_prediction_nn(pred, y):

    # Configuration
    yshp = np.shape(y)
    m = yshp[0]
    k = yshp[1]
    accuracy = np.sum(np.argmax(pred,axis=1) == np.argmax(y,axis=1)) / m

    # Compute Fscore
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for i in range(m):
        if ((y[i,1]==1) & (pred[i,1]>=0.5)):
            tp += 1
        if ((y[i,0]==1) & (pred[i,1]>=0.5)):
            fp += 1
        if ((y[i,0]==1) & (pred[i,0]>=0.5)):
            tn += 1
        if ((y[i,1]==1) & (pred[i,0]>=0.5)):
            fn += 1

    # Check
    accuracy = (tp+tn)/m
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    fscore = 2 / ((1/precision)+(1/recall))

    return precision, recall, fscore

def pca(x,t):

    xshp = np.shape(x)
    m = xshp[0]
    d = xshp[1]

    # Matrix
    X = (1/m)*np.matmul(x.T,x)

    # Eigen decomposition
    w, u = np.linalg.eig(X)

    # Display eigenvalues
    print('Eigenvalues = ',w)
    
    # Pick k values
    max_w = np.max(w)
    idx = np.where(w >= t*max_w)
    kmax = len(idx[0])
    print('Number of principal components = ',kmax)

    # Pick principal components
    wp = np.zeros((kmax,1))
    up = np.zeros((kmax,d))
    for k in range(kmax):
        wp[k] = w[idx[0][k]]
        up[k,:] = u[idx[0][k],:]
        
    xpca = np.matmul(x,up.T)
    print('xpca size = ',np.shape(xpca))

    return w, u, xpca
    

    

    
