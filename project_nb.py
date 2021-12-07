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
    x, y = project_util.load_dataset(train_path, add_intercept=False)

    # Check
    xshp = np.shape(x)
    yshp = np.shape(y)
    m = xshp[0]
    d = xshp[1]
    print('Original data : x shape = ',xshp,' : y shape = ',yshp)

    # Compute y statistics and create a logistic problem
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
            
    # Setup Naive Bayes
    clf = NaiveBayes()
    
    # Compute parameters of Naive Bayes classifier
    clf.compute(x_train, y_train)

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


class NaiveBayes:

    def __init__(self, d=30):

        self.mu0 = np.zeros((1,d))
        self.mu1 = np.zeros((1,d))
        self.sigma0 = np.identity(d)
        self.sigma1 = np.identity(d)


    def compute(self, x, y):

        # Get dimensions
        shx = np.shape(x)
        m = shx[0]
        d = shx[1]

        # Initialize
        self.mu0 = np.zeros((1,d))
        self.mu1 = np.zeros((1,d))
        self.sigma0 = np.identity(d)
        self.sigma1 = np.identity(d)

        # Create clean matrices
        ynew = np.zeros((m,1))
        xnew = np.zeros((m,d))
        for k in range(m):
            ynew[k] = y[k]
            xnew[k,:] = x[k,:]
        y = ynew
        x = xnew
        
        # Label indices
        pos_idx = np.where(y==1)
        neg_idx = np.where(y==0)
        num_pos = len(pos_idx[0])
        num_neg = len(neg_idx[0])

        # Mu0
        x0 = x[neg_idx[0],:]
        self.mu0 = x0.mean(0)

        # Mu1
        x1 = x[pos_idx[0],:]
        self.mu1 = x1.mean(0)

        # Sigma0
        xx0 = x0-self.mu0
        for i in range(num_neg):
            self.sigma0 += (1/num_neg) * np.outer(xx0[i,:],xx0[i,:])

        # Sigma1
        xx1 = x1-self.mu1
        for i in range(num_pos):
            self.sigma1 += (1/num_pos) * np.outer(xx1[i,:],xx1[i,:])

        print('Naive Bayes')
        print('Done computing mu0, mu1, sigma0, sigma1')
        print('mu0 = ',self.mu0)
        print('mu1 = ',self.mu1)
        print('')

    def predict(self, x):
        
        # Get dimensions
        shx = np.shape(x)
        m = shx[0]
        d = shx[1]

        # Initialize
        mu0 = self.mu0
        mu1 = self.mu1
        sigma0 = self.sigma0
        sigma1 = self.sigma1
        pred = np.zeros((m,1))

        # Prediction
        det0 = np.linalg.det(sigma0)
        det1 = np.linalg.det(sigma1)
        sinv0 = np.linalg.inv(sigma0)
        sinv1 = np.linalg.inv(sigma1)
        for i in range(m):
            xx = x[i,:]-mu0
            val0 = np.matmul(xx,np.matmul(sinv0,xx.T))
            val0 = (1/np.sqrt(det0))*np.exp(-0.5*val0)
            xx = x[i,:]-mu1
            val1 = np.matmul(xx,np.matmul(sinv1,xx.T))
            val1 = (1/np.sqrt(det1))*np.exp(-0.5*val1)
            if (val1 > val0):
                pred[i] = 1

        return pred


if __name__ == '__main__':
    main(train_path='data_train.csv',
         save_path='nb_pred.txt')
