import numpy as np
from scipy.special import logsumexp
from scipy.special import expit as logistic_sigmoid

from matplotlib import pyplot as plt

from LRGradientDescent import LogisticRegressionGradientDescent

import sklearn.preprocessing
import sklearn.pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class LRGDWithFeatureTransform(LogisticRegressionGradientDescent):
    ''' Logistic Regression classifier using special feature transform preprocessing!

    Implements the standard sklearn binary classifier API:
    * fit : train the model and set internal trainable attributes
    * predict : produce hard binary predictions
    * predict_proba : produce probabilistic predictions for both labels (0 and 1)

    Inherits all methods from LogisticRegressionGradientDescent, including:
    * constructor
    * gradient descent training of weights via fit
    * utility methods like initialize_w

    '''

    def fit_feature_pipeline(self, x_NF, y_N=None):
        ''' Train the feature-processing pipeline that happens before the LR classifier

        Args
        ----
        x_NF : 2D array, size N x F (n_examples x n_features)
            Each row is a feature vector
        y_N : 1D array, size N (n_examples)
            Each entry is a binary label

        Returns
        -------
        Nothing. Internal attribute 'feature_transform_pipeline' updated.
        '''
        # TODO edit this pipeline!
        # Add whatever additional preprocessing would be useful!
        self.feature_transform_pipeline = sklearn.pipeline.Pipeline(
            [('feature_transform', sklearn.pipeline.FeatureUnion(transformer_list=[
                    ('original_x', sklearn.preprocessing.PolynomialFeatures(degree=1, include_bias=False)),
                    # ('squared_x', SquaredFeatureExtractor()),
                    #('avg_x', AverageValueFeatureExtractor()),
                    ]),
            )])
        self.feature_transform_pipeline.fit(x_NF, y_N)

    def predict_proba(self, x_NF):
        ''' Produce soft probabilistic predictions for provided input features

        Args
        ----
        x_NF : 2D array, size N x F (n_examples x n_features_excluding_bias)
            Input features (one row per example).

        Returns
        -------
        yproba_N2 : 2D array, size N x 2
            First column gives probability of zero label (negative)
            Second column gives probability of one label (positive)
            Each entry is a non-negative probability value within (0.0, 1.0)
            Each row sums to one
        '''
        # Transform each original F-dim feature vector to a new H-dim vector!
        feat_NH = self.feature_transform_pipeline.transform(x_NF)
        # Now call your original predict_proba code defined in parent class
        # but passing in the transformed feats!
        return super(LRGDWithFeatureTransform, self).predict_proba(feat_NH)

    ### Method for training

    def fit(self, x_NF, y_N):
        ''' Fit logistic regression model to provided training data

        Will minimize the loss function defined by calc_loss

        Returns
        -------
        Nothing. Only internal instance attributes updated.

        Post Condition
        --------------
        Internal attributes are updated:
        * w_G contains the optimal weights
        * trace_loss contains loss at every step of gradient descent
        * trace_L1_norm_of_grad contains L1 norm of grad after every step
        '''
        # Train the feature processing pipeline
        self.fit_feature_pipeline(x_NF, y_N)
        # Transform each original F-dim feature vector to a new H-dim vector!
        feat_NH = self.feature_transform_pipeline.transform(x_NF)
        ## Now call your original gradient descent code,
        # as defined in the parent class LogisticRegressionGradientDescent,
        # but use the transformed features as input to LR classifier 
        return super(LRGDWithFeatureTransform, self).fit(feat_NH, y_N)


class SquaredFeatureExtractor(BaseEstimator, TransformerMixin):
    """ Extracts feature equal to square of each original feature
    """

    def __init__(self):
        pass

    def transform(self, x, y=None):
        """ Average all feature values into a new feature column

        Args
        ----
        x : 2D array, size F

        Returns
        -------
        feat : 2D array, size N x F
            One feature extracted for each example
        """
        return np.square(x)

    def fit(self, x, y=None):
        """ Nothing happens when fitting
        """
        return self

class AverageValueFeatureExtractor(BaseEstimator, TransformerMixin):
    """ Extracts feature equal to the *sum* of all pixels in image
    """

    def __init__(self):
        pass

    def transform(self, x, y=None):
        """ Average all feature values into a new feature column

        Returns
        -------
        feat : 2D array, size N x 1
            One feature extracted for each example
        """
        return np.sum(x, axis=1)[:,np.newaxis]

    def fit(self, x, y=None):
        """ Nothing happens when fitting
        """
        return self



if __name__ == '__main__':
    ## Toy problem 
    #
    # Generate negative-label points in a tight cluster centered at zero
    # General positive-label points in a wide ring centered at zero
    #
    # Logistic regression on original features should struggle
    # But using our *squared* features, we should get better performance


    # C = 50 # num examples per class
    # N = 2 * C # num examples total
    # random_state = np.random.RandomState(0)
    # x_pos_C2 = 0.5 * random_state.randn(C, 2)
    # x_neg_M2 = 5.0 * random_state.randn(100 * C, 2)
    # radius_M = np.sqrt(np.sum(np.square(x_neg_M2), axis=1))
    # fits_ring_M = np.logical_and(radius_M >= 1.8, radius_M <= 2.5)
    # x_neg_C2 = x_neg_M2[fits_ring_M][:C]

    x_N2 = np.loadtxt('data_sneaker_vs_sandal/x_train.csv', delimiter=',', skiprows=1)
    y_N = np.loadtxt('data_sneaker_vs_sandal/y_train.csv', delimiter=',', skiprows=1)

    feature_avg_lum = []
    loc_lum = []
    num = 0
    for i in x_N2:
        for x in np.where(i > 0)[0]:
            loc_lum.append(i[x])
        num = num + 1
        print(num)c
        feature_avg_lum.append(np.average(i))
    print(feature_avg_lum)

    #plt.plot(x_N2[y_N == 1,0], x_N2[y_N == 1,1], color='b', marker='+', ls='')
    #plt.plot(x_N2[y_N == 0,0], x_N2[y_N == 0,1], color='r', marker='x', ls='')

    ## Run LR on original features!
    # orig_lr = LogisticRegressionGradientDescent(alpha=100.0, step_size=0.1)
    # orig_lr.fit(x_N2, y_N)

    ## Run LR on transformed features!
    new_lr = LRGDWithFeatureTransform(alpha=1, step_size=0.001)
    new_lr.fit(x_N2, y_N)

    np.set_printoptions(precision=4, suppress=1, linewidth=100)
    for nn in range(3):
        print("Example %d" % nn)
        print("Original feature vector: %s" % x_N2[nn])
        print("     New feature vector: %s" % (
            new_lr.feature_transform_pipeline.transform(x_N2[nn:nn+1])[0]))

    yproba1_orig_N = orig_lr.predict_proba(x_N2)[:,1]
    yproba1_new_N = new_lr.predict_proba(x_N2)[:,1]

    # print("Training Error with orig features: %.3f" % (
        # np.mean(np.logical_xor(y_N, yproba1_orig_N >= 0.5))))
    print("Training Error with new  features: %.3f" % (
        np.mean(np.logical_xor(y_N, yproba1_new_N >= 0.5))))

    #plt.show()
