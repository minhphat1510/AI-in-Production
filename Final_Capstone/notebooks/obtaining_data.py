#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 19:38:55 2021

@author: mouadzeghraoui
"""
import os
import pandas as pd
import json
import numpy as np

def obtaining_data(path_to_json):
    '''
    Reads all the json files (my training data).
    Returns my training data as a dataframe
    '''
    column_dict = {'country': 'country', 'customer_id': 'customer_id','invoice': 'invoice',
                       'total_price': 'price', 'StreamID': 'stream_id',
                        'TimesViewed': 'times_viewed'}
    wrong_columns = ['TimesViewed', 'StreamID', 'total_price']
    columns_name = ['country', 'customer_id', 'invoice', 'price', 'stream_id', 'times_viewed']
    json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
    json_files = sorted(json_files)
    df_final = pd.DataFrame(columns = columns_name)
    
    # we need both the json and an index number so use enumerate()
    for index, js in enumerate(json_files):
        with open(os.path.join(path_to_json, js)) as json_file:
            json_text = json.load(json_file)
            df = pd.DataFrame(json_text)
        for i in wrong_columns:
            if i in df.columns:
                df = df.rename(columns=column_dict)
                
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
        #My Final table
        df_final = df_final.append(df, ignore_index=True)
    return df_final




def correct_datatype(df):
        '''
        Maintain Datatypes.
        '''
        df = df.astype({ "country": str,"price": float,"times_viewed": int})
        return df
    
# Function to calculate missing values by column
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * mis_val / len(df)
        
        # Make a table with the resul
        
        
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns

def remove_duplicated_rows(df):
    duplicate_rows = df.duplicated()
    if True in duplicate_rows:
        df = df[~duplicate_rows]
    print("Removed {} duplicate rows".format(np.where(duplicate_rows==True)[0].size))
    return df




    