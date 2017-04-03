import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        #raise NotImplementedError

        word_sequences = self.sequences
        best_hmm_model = None

        #n = self.min_n_components
        feature_cnt = self.X.shape[1]
        #print(feature_cnt)

        best_BIC_Score =float("inf")

        for num_states in  range(self.min_n_components, self.max_n_components+1):


            try:


                # train a model based on current number of components
             hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
             # calculate log loss for the model

             logL = hmm_model.score(self.X, self.lengths)

            # compute the number of parameters in the model

             #p = num_states * (num_states - 1) * 2 * feature_cnt * num_states
             p = num_states * (feature_cnt * 2 + 1)
             logN = np.log(len(self.X))
             #print(len(self.X)) # size of data
                # Calculate BIC score using provided calculation and above model parameters

             BIC_Score = -2 * logL + p * logN
            #BIC_Score =-2*np.log(len(self.X)) + p*
             #print("BIC for %d components is %d" % (num_states,BIC_Score))
             #for mimiumm score
             if best_BIC_Score>BIC_Score:
                 best_hmm_model=hmm_model
                 best_BIC_Score=BIC_Score

             #    pass
            #n += 1
            except:
              pass
        #print("best BIC  is %d" % ( best_BIC_Score))

        return best_hmm_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_DIC_Score = float('-inf')
        M = len((self.words).keys()) #num of words
        #M = float('inf') if len((self.words).keys()) == 1 else len((self.words).keys())

        best_hmm_model = None

        for num_states in range(self.min_n_components, self.max_n_components + 1):

            try:
                hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)

                LogLi = hmm_model.score(self.X, self.lengths)

            except:
                pass

            SumLogL = 0

            for each_word in self.hwords.keys():


                X_each_word, lengths_each_word = self.hwords[each_word]

            try:
                SumLogL += hmm_model.score(X_each_word, lengths_each_word)
                 #each_hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                 #                      random_state=self.random_state, verbose=False).fit(X_each_word,lengths_each_word)
                #SumLogL += hmm_model.score(X_each_word, lengths_each_word)
            except:
                SumLogL += 0







            #DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
            DIC_Score = LogLi - (1 / (M - 1)) * (SumLogL-LogLi)
            #DIC_Score = LogLi - (1 / (M - 1)) * (LogLSum)
            #print("DIC for %d components is %d" % (num_states, DIC_Score))
            #get maximizing score
            if DIC_Score> best_DIC_Score:
                best_DIC_Score = DIC_Score
                best_hmm_model = hmm_model
        #print("best DIC  is %d" % (best_DIC_Score))
        return best_hmm_model



class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV

        best_CV_Score = float('-inf')

        best_hmm_model = None


        '''
        >>> from sklearn.model_selection import KFold
>>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
>>> y = np.array([1, 2, 3, 4])
>>> kf = KFold(n_splits=2)
>>> kf.get_n_splits(X)
2
>>> print(kf)
KFold(n_splits=2, random_state=None, shuffle=False)
>>> for train_index, test_index in kf.split(X):
...    print("TRAIN:", train_index, "TEST:", test_index)
...    X_train, X_test = X[train_index], X[test_index]
...    y_train, y_test = y[train_index], y[test_index]
TRAIN: [2 3] TEST: [0 1]
TRAIN: [0 1] TEST: [2 3]
        '''
        #print(self.X)
        #print(self.lengths)
        split_method = KFold(n_splits=2)

        for num_states in range(self.min_n_components, self.max_n_components + 1):

            SumLogL = 0
            Counter = 1

            #for cv_train, cv_test in split_method.split(self.sequences):
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                #print("Train fold indices:{} Test fold indices:{}".format(cv_train_idx, cv_train_idx))


                X_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)

                try:

                    hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(X_train, lengths_train)

                    LogL = hmm_model.score(X_test, lengths_test)

                    Counter += 1

                except:
                    pass

                SumLogL+= LogL

            CV_Score = SumLogL / (Counter+1.0)
            #print("CV for %d components is %d" % (num_states, CV_Score))

            if CV_Score>best_CV_Score:
                best_CV_Score = CV_Score
                best_hmm_model = hmm_model

        return best_hmm_model


