import pandas as pd


class Preprocessor:
    """
    Preprocesses data. Takes data from csv, cleans data and prepares it for division into subpopulations.
    """

    def __init__(self, consts):
        self.consts = consts

    def upload_dataset(self):
        """
        Uploads dataset from csv file.

        :return: Dataframe of dataset.
        """
        df = pd.read_csv(self.consts['file'])
        df = df.drop(self.consts['drop_columns'], axis=1)  # Remove columns that are deemed irrelevant
        df = df[~df.isin(self.consts['null_values']).any(axis=1)]  # Remove all rows that contain any null values
        return df

    @staticmethod
    def unify_similar_data(df, col, category, new_value, continuous):
        """
        Unifies similar data in a column. We want data to be unified in order to reduce the number of categories.

        :param df: Dataframe to unify data in.
        :param col: Column to unify data in.
        :param category: List of values to unify (categorical) or upper bound (continuous).
        :param new_value: Value to unify data to (categorical) or lower bound (continuous).
        :param continuous: Describes whether columns is .
        """
        def list_unification(cat, new_val):  # Unifies data based on list of categories
            return lambda x: new_val if x in cat else x

        def continuous_unification(upper, lower):  # Unifies data based on upper and lower bounds
            return lambda x: lower if lower <= x < upper else x

        unification = continuous_unification if continuous else list_unification

        df[col] = df[col].apply(unification(category, new_value))
        return df

    def prepare_division(self, df):
        """
        Prepares division of df into subpopulations. Decreases number of categories in columns based on BINS.
        """
        for col in self.consts['bins'].keys():  # Bins are either groups of categories or intervals, depending on type
            if type(self.consts['bins'][col][0]) == list:  # Bins are groups (lists) of categories
                for i in range(len(self.consts['bins'][col])):
                    new_val = self.consts['new_names'][col][i]  # Rename columns whose values are in list to new names
                    df = self.unify_similar_data(df, col, self.consts['bins'][col][i], new_val, False)
            else:  # The intervals are the numbers between the numbers in the list.
                for i in range(len(self.consts['bins'][col])):
                    lower_bound = self.consts['bins'][col][i-1] if i != 0 else df[col].min()
                    df = self.unify_similar_data(df, col, self.consts['bins'][col][i], lower_bound, True)
        return df

    def divide_to_subpops(self, df):
        """
        Divides df into subpopulations. Each subpopulation is a dataframe.

        :param df: Dataframe to divide into subpopulations.
        :return: List of dataframes, each dataframe is a subpopulation.
        """
        subpops = [df]
        for col in self.consts['subpops']:
            values = df[col].unique()
            subpops = [subpop[subpop[col] == value] for value in values for subpop in subpops]  # For each column,
            # divide each subpopulation into subpopulations
        return subpops

    def interpret_data(self):
        df = self.upload_dataset()
        df = self.prepare_division(df)
        return self.divide_to_subpops(df)
