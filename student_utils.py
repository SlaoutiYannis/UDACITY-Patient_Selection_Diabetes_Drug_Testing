import pandas as pd
import numpy as np
import os
import tensorflow as tf

from functools import partial  # used and explained in question 8

####### STUDENTS FILL THIS OUT ######
#Question 3
def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''
    def check_if_empty(ndc):
        if ndc.empty:
            return np.nan
        else:
            return ndc.values[0]

    df['generic_drug_name'] = [check_if_empty(ndc) for ndc in [ndc_df[ndc_df["NDC_Code"]==code]["Non-proprietary Name"]
                                                               for code in df["ndc_code"]]]
    df = df.drop(['ndc_code'], axis=1)
    return df


#Question 4
def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''
    first_encounter_ids = [min(df[df['patient_nbr']==nbr]['encounter_id']) for nbr in set(df['patient_nbr'])]
    first_encounter_df = df[df['encounter_id'].isin(first_encounter_ids)]
    return first_encounter_df


#Question 6
def patient_dataset_splitter(df, patient_key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    df = df.iloc[np.random.permutation(len(df))]
    unique_values = df[patient_key].unique()
    total_values = len(unique_values)
    sample_size_1 = round(total_values * (0.6))
    sample_size_2 = round(total_values * (0.8))
    train = df[df[patient_key].isin(unique_values[:sample_size_1])].reset_index(drop=True)
    test = df[df[patient_key].isin(unique_values[sample_size_1:sample_size_2])].reset_index(drop=True)
    validation = df[df[patient_key].isin(unique_values[sample_size_2:])].reset_index(drop=True)

    print("Training partition has a shape = ", train.shape) 
    print("Test partition has a shape = ", test.shape)
    print("Validation partition has a shape = ", validation.shape)
    
    return train, validation, test


#Question 7

def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        '''
        Which TF function allows you to read from a text file and create a categorical feature
        You can use a pattern like this below...
        tf_categorical_feature_column = tf.feature_column.......

        '''
        tf_categorical_feature_column = tf.feature_column.categorical_column_with_vocabulary_file(key=c, vocabulary_file=vocab_file_path)
        tf_categorical_feature_column = tf.feature_column.indicator_column(tf_categorical_feature_column)
        output_tf_list.append(tf_categorical_feature_column)
    return output_tf_list


#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std



def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    # used the functools.partial func as per https://knowledge.udacity.com/questions/189045
    # I made sure to inderstand it, very useful !
    normalizer = partial(normalize_numeric_with_zscore, mean = MEAN, std = STD)
    tf_numeric_feature = tf.feature_column.numeric_column(key=col,  dtype=tf.float64, default_value=default_value,
                                                          normalizer_fn=normalizer)
    return tf_numeric_feature


#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()
    return m, s


# Question 10
def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    student_binary_prediction = np.array([int(number>5) for number in df[col]])
    return student_binary_prediction
