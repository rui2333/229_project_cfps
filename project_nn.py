import numpy as np
import matplotlib.pyplot as plt
import argparse
import project_util

def softmax(x):

    # Compute softmax per column
    nxp = np.shape(x)
    y = np.zeros(nxp)
    for i in range(nxp[1]):
        gen = np.exp(x[:,i])
        y[:,i] = gen / np.sum(gen)

    return y

def sigmoid(x):

    y = 1./(1+np.exp(-x))
    
    return y


def get_initial_params(input_size, num_hidden, num_output):
    """
    Compute the initial parameters for the neural network.

    This function should return a dictionary mapping parameter names to numpy arrays containing
    the initial values for those parameters.

    There should be four parameters for this model:
    W1 is the weight matrix for the hidden layer of size input_size x num_hidden
    b1 is the bias vector for the hidden layer of size num_hidden
    W2 is the weight matrix for the output layers of size num_hidden x num_output
    b2 is the bias vector for the output layer of size num_output

    As specified in the PDF, weight matrices should be initialized with a random normal distribution
    centered on zero and with scale 1.
    Bias vectors should be initialized with zero.

    Args:
        input_size: The size of the input data
        num_hidden: The number of hidden states
        num_output: The number of output classes

    Returns:
        A dict mapping parameter names to numpy arrays
    """

    d = input_size
    h = num_hidden
    K = num_output

    # Initialize weights and biases
    W1 = np.random.randn(d,h)
    b1 = np.zeros((h,1))
    W2 = np.random.randn(h,K)
    b2 = np.zeros((K,1))
    dW1 = np.zeros((d,h))
    db1 = np.zeros((h,1))
    dW2 = np.zeros((h,K))
    db2 = np.zeros((K,1))

    # Create parameters
    params = {'W1': W1, 'b1': b1, 'W2':W2, 'b2': b2, 'dW1': dW1, 'db1': db1, 'dW2':dW2, 'db2': db2}

    return params

    # *** END CODE HERE ***

def forward_prop(data, labels, params):
    """
    Implement the forward layer given the data, labels, and params.

    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network

    Returns:
        A 3 element tuple containing:
            1. A numpy array of the activations (after the sigmoid) of the hidden layer
            2. A numpy array The output (after the softmax) of the output layer
            3. The average loss for these data elements
    """

    # Extract params
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
    X = np.transpose(data)
    Y = np.transpose(labels)
    
    # Forward propagation
    Z1 = np.matmul(np.transpose(W1),X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.matmul(np.transpose(W2),A1) + b2
    A2 = softmax(Z2)

    # Add new params
    params['Z1'] = Z1
    params['A1'] = A1
    params['Z2'] = Z2
    params['A2'] = A2

    # Cost
    nxp = np.shape(A2)
    n = nxp[1]
    cost = 0
    for j in range(nxp[1]):
        cost -= np.sum(Y[:,j] * np.log(A2[:,j]))
    cost /= n

    h = A1
    output = np.transpose(A2)

    return h, output, cost
    

def backward_prop(data, labels, params, forward_prop_func):
    """
    Implement the backward propegation gradient computation step for a neural network

    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network
        forward_prop_func: A function that follows the forward_prop API above

    Returns:
        A dictionary of strings to numpy arrays where each key represents the name of a weight
        and the values represent the gradient of the loss with respect to that weight.

        In particular, it should have 4 elements:
            W1, W2, b1, and b2
    """

    nxd = np.shape(data)

    # Extract params
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
    Z1 = params['Z1']
    A1 = params['A1']
    Z2 = params['Z2']
    A2 = params['A2']
    X = np.transpose(data)
    Y = np.transpose(labels)
    n = nxd[0]

    # Back Propagation
    dZ2 = A2 - Y
    dW2 = (1/n) * np.matmul(A1,np.transpose(dZ2))
    db2 = (1/n) * np.sum(dZ2, axis=1, keepdims=True)
    dA1 = np.matmul(W2,dZ2)
    dZ1 = dA1 * np.multiply(sigmoid(Z1),(1-sigmoid(Z1)))
    dW1 = (1/n) * np.matmul(X,np.transpose(dZ1))
    db1 = (1/n) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2


def backward_prop_regularized(data, labels, params, forward_prop_func, reg):
    """
    Implement the backward propagation gradient computation step for a neural network

    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network
        forward_prop_func: A function that follows the forward_prop API above
        reg: The regularization strength (lambda)

    Returns:
        A dictionary of strings to numpy arrays where each key represents the name of a weight
        and the values represent the gradient of the loss with respect to that weight.

        In particular, it should have 4 elements:
            W1, W2, b1, and b2
    """

    nxd = np.shape(data)

    # Extract params
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
    Z1 = params['Z1']
    A1 = params['A1']
    Z2 = params['Z2']
    A2 = params['A2']
    X = np.transpose(data)
    Y = np.transpose(labels)
    n = nxd[0]

    # Back Propagation
    dZ2 = A2 - Y
    dW2 = (1/n) * np.matmul(A1,np.transpose(dZ2))
    db2 = (1/n) * np.sum(dZ2, axis=1, keepdims=True)
    dA1 = np.matmul(W2,dZ2)
    dZ1 = dA1 * np.multiply(sigmoid(Z1),(1-sigmoid(Z1)))
    dW1 = (1/n) * np.matmul(X,np.transpose(dZ1))
    db1 = (1/n) * np.sum(dZ1, axis=1, keepdims=True)

    # Add Regularized terms
    dW2 += (reg*2*W2)
    dW1 += (reg*2*W1)

    return dW1, db1, dW2, db2

def gradient_descent_epoch(train_data, train_labels, learning_rate, batch_size, params, forward_prop_func, backward_prop_func):
    """
    Perform one epoch of gradient descent on the given training data using the provided learning rate.

    This code should update the parameters stored in params.
    It should not return anything

    Args:
        train_data: A numpy array containing the training data
        train_labels: A numpy array containing the training labels
        learning_rate: The learning rate
        batch_size: The amount of items to process in each batch
        params: A dict of parameter names to parameter values that should be updated.
        forward_prop_func: A function that follows the forward_prop API
        backward_prop_func: A function that follows the backwards_prop API

    Returns: This function returns nothing.
    """

    dxp = np.shape(train_data)
    lxp = np.shape(train_labels)
    n = dxp[0]
    B = batch_size
    num_batches = np.uint(n/B)


    # Batch Training
    for i in range(num_batches):
        
        batch_data = train_data[(i*B):((i+1)*B),:]
        batch_labels = train_labels[(i*B):((i+1)*B),:]
        
        h, output, cost = forward_prop_func(batch_data, batch_labels, params)
        dW1, db1, dW2, db2 = backward_prop_func(batch_data, batch_labels, params, forward_prop_func)

        # Update params
        params['W1'] -= (learning_rate * dW1)
        params['b1'] -= (learning_rate * db1)
        params['W2'] -= (learning_rate * dW2)
        params['b2'] -= (learning_rate * db2)

    print('Epoch complete - Cost = ',cost)

    # This function does not return anything
    return

def nn_train(train_data, train_labels, dev_data, dev_labels, get_initial_params_func, forward_prop_func, backward_prop_func,
            num_hidden=30, learning_rate=5, num_epochs=50, batch_size=64):

    # Read input and output dimensions
    (m, d) = train_data.shape
    (m, k) = train_labels.shape

    # Initialize Parameters
    params = get_initial_params_func(d, num_hidden, k)

    # Epoch loop
    cost_train = []
    cost_dev = []
    accuracy_train = []
    accuracy_dev = []
    for epoch in range(num_epochs):
        gradient_descent_epoch(train_data, train_labels, learning_rate, batch_size, params, forward_prop_func, backward_prop_func)
        h, output, cost = forward_prop_func(train_data, train_labels, params)
        cost_train.append(cost)
        accuracy_train.append(compute_accuracy(output,train_labels))
        h, output, cost = forward_prop_func(dev_data, dev_labels, params)
        cost_dev.append(cost)
        accuracy_dev.append(compute_accuracy(output, dev_labels))

    return params, cost_train, cost_dev, accuracy_train, accuracy_dev

def nn_test(data, labels, params):
    h, output, cost = forward_prop(data, labels, params)
    accuracy = compute_accuracy(output, labels)
    precision, recall, fscore = project_util.analyze_prediction_nn(output,labels)
    
    return accuracy, precision, recall, fscore

def compute_accuracy(output, labels):
    accuracy = (np.argmax(output,axis=1) == np.argmax(labels,axis=1)).sum() * 1. / labels.shape[0]
    return accuracy
    
def run_train_test(name, all_data, all_labels, backward_prop_func, num_epochs):
    
    params, cost_train, cost_dev, accuracy_train, accuracy_dev = nn_train(all_data['train'], all_labels['train'], all_data['dev'], all_labels['dev'],
                                                                        get_initial_params, forward_prop, backward_prop_func,
                                                                        num_hidden=30, learning_rate=5, num_epochs=num_epochs, batch_size=64)

    t = np.arange(num_epochs)

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(t, cost_train,'r', label='train')
    ax1.plot(t, cost_dev, 'b', label='dev')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    if name == 'baseline':
        ax1.set_title('Without Regularization')
    else:
        ax1.set_title('With Regularization')
    ax1.legend()

    ax2.plot(t, accuracy_train,'r', label='train')
    ax2.plot(t, accuracy_dev, 'b', label='dev')
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('accuracy')
    ax2.legend()

    fig.savefig('./' + name + '.pdf')

    
    accuracy, precision, recall, fscore = nn_test(all_data['train'], all_labels['train'], params)
    print('Training -- Model = ',name)
    print('Accuracy = ',accuracy)
    print('Precision = ',precision,' : Recall = ',recall,' : Fscore = ',fscore)
    print('')
    accuracy, precision, recall, fscore = nn_test(all_data['dev'], all_labels['dev'], params)
    print('Validation -- Model = ',name)
    print('Accuracy = ',accuracy)
    print('Precision = ',precision,' : Recall = ',recall,' : Fscore = ',fscore)
    print('')
    accuracy, precision, recall, fscore = nn_test(all_data['test'], all_labels['test'], params)
    print('Test -- Model = ',name)
    print('Accuracy = ',accuracy)
    print('Precision = ',precision,' : Recall = ',recall,' : Fscore = ',fscore)
    print('')

    return accuracy

def main(train_path,save_path):

    parser = argparse.ArgumentParser(description='Train a nn model.')
    parser.add_argument('--num_epochs', type=int, default=100)
    args = parser.parse_args()

    # Load data
    x, y = project_util.load_dataset(train_path, add_intercept=False)

    # Check
    xshp = np.shape(x)
    yshp = np.shape(y)
    m = xshp[0]
    d = xshp[1]
    print('Original data : x shape = ',xshp,' : y shape = ',yshp)

    # Compute y statistics and create a NN problem
    std_thresh = +1.5
    xnew, ynew = project_util.create_nn(x,y,std_thresh)

    # Normalize
    xnew = project_util.normalize(xnew)

    # PCA analysis
    pca_threshold = 0.01
    w, u, xpca = project_util.pca(xnew,pca_threshold)

    # Re-calibrate
    xshp = np.shape(xpca)
    yshp = np.shape(ynew)
    m = xshp[0]
    d = xshp[1]

    # Training, Validation, Test Set Sizes
    m_train = np.uint(0.7*m)
    m_valid = np.uint(0.1*m)
    m_test = m - (m_train + m_valid)

    # Create Train, Validation and Test Sets
    x_train, y_train, x_valid, y_valid, x_test, y_test = project_util.create_sets_nn(xpca, ynew, m_train, m_valid, m_test)

    # Compile data
    all_data = {'train': x_train,'dev': x_valid,'test': x_test}
    all_labels = {'train': y_train,'dev': y_valid,'test': y_test}

    baseline_acc = run_train_test('baseline', all_data, all_labels, backward_prop, args.num_epochs)
    reg_acc = run_train_test('regularized', all_data, all_labels, lambda a, b, c, d: backward_prop_regularized(a, b, c, d, reg=0.0001),
                             args.num_epochs)

    return baseline_acc, reg_acc

if __name__ == '__main__':
    main(train_path='data_train.csv',
         save_path='nn_pred.txt')
