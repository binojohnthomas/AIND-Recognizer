# TODO Choose a feature set and model selector
from my_recognizer import recognize
from asl_utils import show_errors

from my_model_selectors import *

import numpy as np
import pandas as pd
from asl_data import AslDb




asl = AslDb() # initializes the database

df_means = asl.df.groupby('speaker').mean()
df_std = asl.df.groupby('speaker').std()
from asl_utils import test_features_tryit
# TODO add df columns for 'grnd-rx', 'grnd-ly', 'grnd-lx' representing differences between hand and nose locations
asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']
asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']

features_ground = ['grnd-rx','grnd-ry','grnd-lx','grnd-ly']

#calcuate Mean and map it to table
asl.df['left-x-mean']= asl.df['speaker'].map(df_means['left-x'])
asl.df['left-y-mean']= asl.df['speaker'].map(df_means['left-y'])
asl.df['right-x-mean']= asl.df['speaker'].map(df_means['right-x'])
asl.df['right-y-mean']= asl.df['speaker'].map(df_means['right-y'])

#calcuate std deviation and map it to table
asl.df['left-x-std']= asl.df['speaker'].map(df_std['left-x'])
asl.df['left-y-std']= asl.df['speaker'].map(df_std['left-y'])
asl.df['right-x-std']= asl.df['speaker'].map(df_std['right-x'])
asl.df['right-y-std']= asl.df['speaker'].map(df_std['right-y'])

# calucate z score

asl.df['norm-rx']= (asl.df['right-x'] - asl.df['right-x-mean'])/asl.df['right-x-std']
asl.df['norm-ry']= (asl.df['right-y'] - asl.df['right-y-mean'])/asl.df['right-y-std']
asl.df['norm-lx']= (asl.df['left-x']  - asl.df['left-x-mean'])/asl.df['left-x-std']
asl.df['norm-ly']= (asl.df['left-y']  - asl.df['left-y-mean'])/asl.df['left-y-std']

features_norm = ['norm-rx', 'norm-ry', 'norm-lx','norm-ly']

asl.df['polar-rr'] = np.sqrt( np.power(asl.df['grnd-rx'], 2) + np.power(asl.df['grnd-ry'], 2))
asl.df['polar-rtheta'] = np.arctan2(asl.df['grnd-rx'], asl.df['grnd-ry'])
asl.df['polar-lr'] = np.sqrt( np.power(asl.df['grnd-lx'], 2) + np.power(asl.df['grnd-ly'], 2))
asl.df['polar-ltheta'] = np.arctan2(asl.df['grnd-lx'], asl.df['grnd-ly'])


features_polar = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']

asl.df['delta-rx'] = asl.df['right-x'].diff().fillna(0)
asl.df['delta-ry'] = asl.df['right-y'].diff().fillna(0)
asl.df['delta-lx'] = asl.df['left-x'].diff().fillna(0)
asl.df['delta-ly'] = asl.df['left-y'].diff().fillna(0)
features_delta = ['delta-rx', 'delta-ry', 'delta-lx', 'delta-ly']

df_means = asl.df.groupby('speaker').mean()
df_std = asl.df.groupby('speaker').std()

polar_rr_mean = asl.df['speaker'].map(df_means['polar-rr'])
polar_lr_mean = asl.df['speaker'].map(df_means['polar-lr'])
polar_rtheta_mean = asl.df['speaker'].map(df_means['polar-rtheta'])
polar_ltheta_mean = asl.df['speaker'].map(df_means['polar-ltheta'])
polar_rr_std = asl.df['speaker'].map(df_std['polar-rr'])
polar_lr_std = asl.df['speaker'].map(df_std['polar-lr'])
polar_rtheta_std = asl.df['speaker'].map(df_std['polar-rtheta'])
polar_ltheta_std = asl.df['speaker'].map(df_std['polar-ltheta'])

asl.df['polar-rr-norm'] = (asl.df['polar-rr'] - polar_rr_mean) / polar_rr_std
asl.df['polar-lr-norm'] = (asl.df['polar-lr'] - polar_lr_mean) / polar_lr_std
asl.df['polar-rtheta-norm'] = (asl.df['polar-rtheta'] - polar_rtheta_mean) / polar_rtheta_std
asl.df['polar-ltheta-norm'] = (asl.df['polar-ltheta'] - polar_ltheta_mean) / polar_ltheta_std

features_custom_polar_norm = ['polar-rr-norm', 'polar-rtheta-norm', 'polar-lr-norm', 'polar-ltheta-norm']

# feature ground norm

df_means = asl.df.groupby('speaker').mean()
df_std = asl.df.groupby('speaker').std()

grnd_rx_mean = asl.df['speaker'].map(df_means['grnd-rx'])
grnd_ry_mean = asl.df['speaker'].map(df_means['grnd-ry'])
grnd_lx_mean = asl.df['speaker'].map(df_means['grnd-lx'])
grnd_ly_mean = asl.df['speaker'].map(df_means['grnd-ly'])
grnd_rx_std = asl.df['speaker'].map(df_std['grnd-rx'])
grnd_ry_std = asl.df['speaker'].map(df_std['grnd-ry'])
grnd_lx_std = asl.df['speaker'].map(df_std['grnd-lx'])
grnd_ly_std = asl.df['speaker'].map(df_std['grnd-ly'])

asl.df['grnd_rx-norm'] = (asl.df['grnd-rx'] - grnd_rx_mean) / grnd_rx_std
asl.df['grnd-ry-norm'] = (asl.df['grnd-ry'] - grnd_ry_mean) / grnd_ry_std
asl.df['grnd-lx-norm'] = (asl.df['grnd-lx'] - grnd_lx_mean) / grnd_ly_std
asl.df['grnd-ly-norm'] = (asl.df['grnd-ly'] - grnd_ly_mean) / grnd_ly_std

features_custom_ground_norm = ['grnd_rx-norm', 'grnd-ry-norm', 'grnd-lx-norm', 'grnd-ly-norm']

features_custom_combine_polarnorm_and_ground=features_custom_polar_norm+ features_ground
features_custom_combine_polarnorm_and_groundnorm=features_custom_ground_norm+features_custom_polar_norm
features_custom_combine_polar_and_ground=features_polar+ features_ground
features_custom_combine_polar_and_groundnorm=features_polar+ features_custom_ground_norm
features_custom_combine_all_base_features= features_polar+features_ground+features_delta+features_norm


def train_all_words(features, model_selector):
    training = asl.build_training(features)  # Experiment here with different feature sets defined in part 1
    sequences = training.get_all_sequences()
    Xlengths = training.get_all_Xlengths()
    model_dict = {}
    for word in training.words:
        model = model_selector(sequences, Xlengths, word,
                        n_constant=3).select()
        model_dict[word]=model
    return model_dict

model_selector_list =[SelectorCV,SelectorDIC, SelectorBIC]
#features_list = [features_custom_polar_norm,features_custom_ground_norm,features_custom_combine_polar_and_ground_norm,features_custom_combine_polar_and_ground]
#features_list=[features_polar,features_ground,features_norm,features_delta]
features_list=[features_custom_combine_all_base_features]

for m_selector in  model_selector_list:
    for feature in features_list:
        format_list = [feature, m_selector]
        print("Using feature - {} and model - {} ".format(*format_list))
        models = train_all_words(feature, m_selector)
        test_set = asl.build_test(feature)
        probabilities, guesses = recognize(models, test_set)
        show_errors(guesses, test_set)

#features = features_norm # change as needed
#model_selector = SelectorDIC # change as needed
#models = train_all_words(features, model_selector)
#test_set = asl.build_test(features)
#probabilities, guesses = recognize(models, test_set)
#show_errors(guesses, test_set)

#models = train_all_words(features_ground, SelectorConstant)

#test_set = asl.build_test(features_ground)
#features = features_ground # change as needed
#model_selector = SelectorBIC # change as needed



# TODO Recognize the test set and display the result with the show_errors method

'''

'''


