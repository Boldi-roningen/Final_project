"""
Created on Thursday 1 feb. 2024:
@author: Boldizsár Jékely s5678684
@email: b.l.jekely@student.rug.nl

Script for calculating abundance, species richness and diversity for points based on point counts of birds
Creates a generalized mixed linear model based on the above calculations with the format:
Abundance/Richness/Diversity ~ Treatment + Date since treatment + Point as a random variable
    :param sample_file <.csv> filename with point count data,
    date should be column 1, id column 2 the rest species counts
    :return <.csv> file with abundance, species richness and diversity
    values as well as the other values for modelling and plotting
    :return <.txt> file with the model outputs
"""
#modules to load:
import pandas as pd
import numpy as np
import statsmodels.api as sm
import sys
import getopt
import os

#functions:
def load_data(file_path):
    """
    Loads data from a csv into a pandas dataframe, replaces missing value with 0's for no observations
    :param file_path: <.csv> the data file with the point count values
    :return: <pandas.DataFrame> a pandas dataframe with the point count data
    """
    try:
        data = pd.read_csv(file_path, sep=",", header=[0], skiprows=[0])
        data = data.fillna(0)
        return data
    except FileNotFoundError as fnf_error:
        print(fnf_error)
        print(f'File not found at {file_path}')

def pop_values(data, output):
    """
    Calculates the abundance and species richness and diversity for each point each year
    :param data: <pandas.DataFrame> pandas dataframe with the point count values
    :param output: <pandas.DataFrame> the output dataframe where we store the values
    :return: <pandas.DataFrame> the output dataframe where we store the values
    """
    row_sum = data.sum(axis=1)
    abundance = data.apply(lambda row: (row > 0).sum(), axis=1)
    relative_abundance = data.div(data.sum(axis=1), axis=0)
    relative_abundance = relative_abundance.replace([np.inf, -np.inf, np.nan], 0)
    shannon_index = -np.sum(relative_abundance * np.log(relative_abundance), axis=1)
    output['Abundance'] = row_sum
    output['Species_richness'] = abundance
    output['Diversity_Index'] = round(shannon_index, 2)
    print('Abundance, Species richness and Diversity Index calculated')
    return output


def modify_date(date_column, id_column, id_subtraction_mapping, output):
    """
    Creates a data column, indicating how long the area has been under treatment
    Dictionary needs to be manually uptaded
    :param date_column: <pandas.DataFrame> the column with the date values
    :param id_column: <pandas.DataFrame> the column with the point ids
    :param id_subtraction_mapping: <Dictionary> ids as keys with the value as start of treatment date
    :param output: <pandas.DataFrame> the output dataframe where we store the values
    :return: <pandas.DataFrame> the output dataframe where we store the values
    """
    print(f'calculating number of years from start of treatment.'
        f'Ensure Id-Date Dictionary is up-to-date, current dictionary: \n'
          f'{id_subtraction_mapping}')
    year_column = pd.to_datetime(date_column).dt.year.astype(int)

    # Check the 'id' column and subtract the corresponding values from 'Year'
    output['Treatment_Year'] = year_column - id_column.apply(
        lambda x: next((v for k, v in id_subtraction_mapping.items() if str(x).startswith(k)), 0))
    output['Modified_Year'] = output['Treatment_Year'].clip(lower=0)
    return output

def apply_treatment(treatment_date, output):
    """
    Add a 1 where the date is positive for treatment adds a 0 for year values of 0
    :param treatment_date: <pandas.DataFrame> the number of years since treatment started
    :param output: pandas.DataFrame> the output dataframe where we store the values
    :return: <pandas.DataFrame> the output dataframe where we store the values
    """
    treatment = (treatment_date > 0).astype(int)
    output['Treatment'] = treatment
    print(f'Treatment values calculated')
    return output


def add_plot_column(ids, output):
    """
    Creates a column with the broader plots for the point from the point count data
    :param ids: <pandas.DataFrame> the column containing the plot ids
    :param output: <pandas.DataFrame> the output dataframe where we store the values
    :return: <pandas.DataFrame> the output dataframe where we store the values
    """
    output['Plot'] = ids.str.extract('(\d+)', expand=False)
    print(f'Plot values extracted')
    return output


def fit_glmm(response_variable, fixed_predictor_1, fixed_predictor_2, random_predictor, data):
    """
    Creates a generalized mixed linear model for the point count data
    :param response_variable: <Column> the response variable (Abundance/Species Richness/Diversity)
    :param fixed_predictor_1: <Column> fixed predictor 1
    :param fixed_predictor_2: <Column> fixed predictor 2
    :param random_predictor: <Column> random predictor
    :param data: <pandas.DataFrame>
    :return: <glmm model> the model fitted on the data
    """
    model_formula = f'{response_variable} ~ {fixed_predictor_1} + {fixed_predictor_2}'
    model = sm.MixedLM.from_formula(model_formula, groups=random_predictor, data=data)
    result = model.fit()
    print(f'model fitted')
    return result


def save_to_csv(data, file_path):
    """
    Saves the pandas dataframe with the population values to a .csv
    :param data: <pandas.DataFrame> the dataframe we want to save to a .csv
    :param file_path: <str> the path to where the csv will be created
    """
    data.to_csv(file_path, index=False)
    print(f'Data saved to: {file_path}')


def save_model_summary(model, file_path, model_name):
    """
    Creates a new file if the one called doesn't exist.
    Then saves the model summaries to the file, adding the summary if it doesn't already exist in the file.

    :param model: the model to be saved
    :param file_path: <str> filepath to save the summary
    :param model_name: the name of the model to identify in the file
    """
    file_exists = os.path.exists(file_path)
    if not file_exists:
        with open(file_path, 'w') as file:
            file.write(f'{"="*20} Model Summaries {"="*20}\n\n')
    with open(file_path, 'r') as file:
        existing_content = file.read()
    if f'{"="*20} {model_name} Model {"="*20}' not in existing_content:
        with open(file_path, 'a') as file:
            file.write(f'\n\n{"="*20} {model_name} Model {"="*20}\n\n')
            file.write(str(model.summary()))
            print(f'Model summary for {model_name} saved to: {file_path}')
    else:
        print(f'Model summary for {model_name} already exists in: {file_path}')



def arguments_getopt(arguments_list):
    """
    Assigns in-line arguments to variables; sample_file, results_file, model_output.
    The list of arguments should exclude argv[0], i.e., the script name itself
    :param arguments_list: <list> with 4 arguments.
    :return: <strings> sample_file, results_file, model_output
    """
    sample_file, results_file, model_output = '', '', ''
    opts, args = getopt.getopt(arguments_list, "hs:v:i:",
                               ["sample_file=", "results_file=", "model_output="])
    for opt, arg in opts:
        if opt == "-h":
            print(f'Please provide the following arguments: \n'
                  f'python script.py -s/--sample_file <sample data file> '
                  f'-v/--values_file <output values file> '
                  f'-i/--sorted_ids_file <sorted ids output file> ')
            sys.exit()
        elif opt in ("-s", "--sample_file"):
            sample_file = arg
        elif opt in ("-v", "--results_file"):
            results_file = arg
        elif opt in ("-i", "--model_output_file"):
            model_output = arg

    return sample_file, results_file, model_output

def write_arguments(sample_file, results_file, model_output):
    """
        Simple function to print values assigned to the input arguments and their type
        :param sample_file: str()
        :param results_file: str()
        :param model_output: str()
        :return:
        """
    print(f'Input arguments:\n'
            f'\tSample file: {sample_file}, {type(sample_file)}\n'
            f'\tOutput values file: {results_file}, {type(results_file)}\n'
            f'\tOutput sample ids for the same individual file: {model_output}, {type(model_output)}\n')




if __name__ == '__main__':
    #Hardcoded input unused:
    #sample_file = 'bird_data_summary_2022.csv'
    #results_file = 'pop_overview.csv'
    #model_output = 'models.txt'
    #Command line input
    sample_file, results_file, model_output = arguments_getopt(sys.argv[1:])
    write_arguments(sample_file, results_file, model_output)

    #Main function
    samples = load_data(sample_file)
    print(f'Samples Imported {samples.head()}')
    output = pd.DataFrame()
    output = pop_values(samples.iloc[:, 2:], output)
    output['ID'] = samples.iloc[:, 1]
    #Dictionary for the date modification
    date_modification = {'S13.': 2014, 'S34.': 2020, 'S19.': 2018}
    output = modify_date(samples.iloc[:, 0], samples.iloc[:, 1], date_modification, output)
    output = apply_treatment(output.iloc[:, 4], output)
    output = add_plot_column(output.iloc[:, 3], output)
    print(f'Output for modelling done \n {output.head()}')
    save_to_csv(output, results_file)
    model_abundance = fit_glmm('Abundance', 'Modified_Year', 'Treatment', 'ID', output)
    model_richness = fit_glmm('Species_richness', 'Modified_Year', 'Treatment', 'ID', output)
    model_diversity = fit_glmm('Diversity_Index', 'Modified_Year', 'Treatment', 'ID', output)
    save_model_summary(model_abundance, model_output, 'Abundance')
    save_model_summary(model_richness, model_output, 'Richness')
    save_model_summary(model_diversity, model_output, 'Diversity')
    print(f'Script done')
