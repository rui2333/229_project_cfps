import numpy as np
import project_util
import time

def main(train_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """

    # Load data
    x, y = project_util.load_dataset(train_path, add_intercept=True)

    # Check
    xshp = np.shape(x)
    yshp = np.shape(y)
    m = xshp[0]
    d = xshp[1]
    print('Original data : x shape = ',xshp,' : y shape = ',yshp)

    # Compute y statistics and create a regression problem
    std_thresh = +1.5
    log_flag = 1
    xnew, ynew, y_threshold = project_util.create_reg(x,y,std_thresh,log_flag)

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
    x_train, y_train, x_valid, y_valid, x_test, y_test = project_util.create_sets(xpca, ynew, m_train, m_valid, m_test, y_threshold, log_flag)
            
    # Setup Logistic Regression
    clf = Regression(logistic_flag=log_flag)
    
    # Train a logistic regression classifier
    clf.fit(x_train, y_train)

    # Training Set
    pred_train = clf.predict(x_train)
    accuracy, precision, recall, fscore = project_util.analyze_prediction(pred_train, y_train, y_threshold, log_flag)
    print('Training Set : Accuracy = ',accuracy,' : Precision = ',precision,' : Recall = ',recall,' : Fscore = ',fscore)
        
    # Validation Set
    pred_valid = clf.predict(x_valid)
    accuracy, precision, recall, fscore = project_util.analyze_prediction(pred_valid, y_valid, y_threshold, log_flag)
    print('Validation Set : Accuracy = ',accuracy,' : Precision = ',precision,' : Recall = ',recall,' : Fscore = ',fscore)

    # Test Set
    pred_test = clf.predict(x_test)
    accuracy, precision, recall, fscore = project_util.analyze_prediction(pred_test, y_test, y_threshold, log_flag)
    print('Test Set : Accuracy = ',accuracy,' : Precision = ',precision,' : Recall = ',recall,' : Fscore = ',fscore)


class Regression:
    """Regression
    Example usage:
        > clf = Regression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.05, max_iter=200000, eps=1e-8, alpha=None, logistic_flag=None, theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.alpha = 0.0001
        self.logistic_flag = logistic_flag
        self.verbose = verbose

    def fit(self, x, y):
        """Logistic Regression
        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """

        shx = np.shape(x)
        m = shx[0]
        d = shx[1]
        self.theta = np.zeros((d,1))
        gz = np.zeros((d,1))
        grad = np.zeros((d,1))
        xi = np.zeros((1,d))

        self.theta = np.random.normal(0,0.1,size=(d,1))

        # Create clean matrices
        ynew = np.zeros((m,1))
        xnew = np.zeros((m,d))
        for k in range(m):
            ynew[k] = y[k]
            xnew[k,:] = x[k,:]
        y = ynew
        x = xnew
        
        # Regression
            
        for iter in range(self.max_iter):

            # Compute gradient
            z = np.matmul(x,self.theta)

            # Logistic vs. ReLu Regression
            if (self.logistic_flag == 1):
                gz = 1 / (1 + np.exp(-z))
            else:
                gz = z
                gz[np.where(z<0)] = 0

            # Gradient
            grad = (1/m) * np.matmul(np.transpose(x),(gz-y))

            # Regularize
            grad += (self.alpha * self.theta)

            # Update theta via iterations
            delta_theta = self.step_size * grad
            self.theta -= delta_theta
            
            delta_norm = np.linalg.norm(delta_theta)
            grad_norm = np.linalg.norm(grad)
            if iter%10000 == 0:
                print('iter = ',iter,' : delta norm = ',delta_norm,' : grad norm = ',grad_norm)
            if delta_norm < self.eps:
                break

        print('****');
        print('exit iter = ',iter);
        print('final delta norm = ',delta_norm);    
        print('****');
        

    def predict(self, x):
        """Return predicted probabilities given new inputs x
        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        
        # Variables
        shx = x.shape
        xnew = np.zeros((shx[0],shx[1]))
        for k in range(shx[0]):
            xnew[k,:] = x[k,:]
        x = xnew
        z = np.matmul(x,self.theta)

        # Prediction
        if (self.logistic_flag == 1):
            pred = 1 / (1 + np.exp(-z))
        else:
            pred = z
            pred[np.where(z<0)] = 0

        return pred


if __name__ == '__main__':
    main(train_path='data_train.csv',
         save_path='logreg_pred.txt')
