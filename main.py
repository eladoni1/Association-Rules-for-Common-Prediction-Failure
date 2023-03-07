# Imports
## Base/Default Libraries:
import platform  # To understand whether we are on linux or windows system.
import shutil  # To copy files to the script's directory so we can load it later on.
from typing import Optional  # Just for typing purposes.
import pandas as pd  # For reading the tabular data.
import numpy as np
import efficient_apriori as ea  # Apriori rules
from scipy import stats

## Utilities:
from numbers import Number

import math

## One-Hot Encoder:
from sklearn.preprocessing import OneHotEncoder

import os.path

## Model:
import xgboost as xgb

## Pickle & Saving objects onto memory:
# import pickle

## Testing:

from pathlib import Path

# "Select File" Window, instead of copy-pasting path
from tkinter import *
from tkinter import filedialog

import os

# Saving files (Encoder + ColumnDivider):
import pickle

# Graphs:
from matplotlib import pyplot as plt

# Global to let the script run on both OS's.
DirectorySlash = '/' if platform.system() == "Linux" else '\\'
x: list = []
y: list = []
increasing_value: int = 0


def addValue(a: int, b: Optional[int] = None):
    print("in addValue, a is : ", a)
    global increasing_value
    y.append(a)
    if b is None:
        x.append(increasing_value)
        increasing_value += 1
    else:
        x.append(b)


def showOnGraph():
    global x, y, increasing_value
    print("x and y are : \n", x ,"\nnow y : \n", y)
    plt.plot(x, y)
    plt.show()
    # Used x and y lists, now clear them.
    x = []
    y = []
    increasing_value = 0



def get_filepath() -> str:
    root.deiconify()
    file_path = filedialog.askopenfilename(initialdir="/", title="Select file", filetypes=(("all files", "*.*"),))
    root.withdraw()
    return file_path


def get_file(request_str: str, requested_ext: str):
    extension = ["", ""]
    while len(extension) < 2 or extension[1] != requested_ext:
        print(request_str)

        filepath_to_train = get_filepath()

        slash_split = filepath_to_train.split('/')

        filename = slash_split[-1]

        full_dir = ""
        for directory in slash_split[:-1]:
            full_dir = full_dir + directory + DirectorySlash

        full_dir = full_dir[:-1]

        print(full_dir, os.getcwd())
        if full_dir != os.getcwd():
            shutil.copy(filepath_to_train, os.getcwd())

        extension = filename.split('.')
    return filename


def removeUniqueColumns(data: pd.DataFrame, percentage: float) -> tuple[pd.DataFrame, list]:
    rows_len = len(data.index)
    columns_removed = []
    for c in data.columns:
        unique_values = data[c].nunique()
        if unique_values >= percentage * rows_len:  # Meaning, %percentage of the rows having unique values.
            columns_removed.append(c)
    data = data.drop(columns_removed, axis=1)  # Dropping that data, because it's too unique-y.
    return data, columns_removed


def getNullifiedColumns(data: pd.DataFrame, percentage: float):
    missing_values_percentage = data.isnull().mean()
    nullified_columns = missing_values_percentage[missing_values_percentage >= percentage].index
    return nullified_columns


def get_rid_nullified(data: pd.DataFrame, percentage: float) -> pd.DataFrame:
    drop_these = getNullifiedColumns(data, percentage)
    return data.drop(drop_these, axis="columns")


def get_rid_incrementals(data: pd.DataFrame, percentage: Number = 1) -> pd.DataFrame:
    if percentage > 1:
        percentage = 1
    drop_these = []
    num_rows = len(data.index)
    for c in data.columns:
        if num_rows * percentage == data[c].nunique(dropna=False):
            drop_these.append(c)
    return data.drop(drop_these, axis="columns")


class ColumnDivider:
    def __init__(self, data: pd.DataFrame, target_feature: str, maximum_uniques: int):
        self.numeric_columns: list = data.dtypes[(data.dtypes == "float64") | (data.dtypes == "int64")].index.tolist()
        # maximum_uniques The number of unique values to be considered 'very numerical'.
        self.very_numerical: list = [nc for nc in self.numeric_columns if data[nc].nunique() > maximum_uniques]
        self.categorical_columns: list = [c for c in data.columns if
                                          c not in self.numeric_columns]
        self.ordinals: list = list(set(self.numeric_columns) - set(self.very_numerical))
        self.numeric_columns.remove(target_feature)
        self.columns_to_ignore: Optional[list[str]] = None

    def get_vn(self):
        return self.very_numerical  # Returns all the columns that are very numerical (more than 10 unique values).

    def get_cat(self):
        return self.categorical_columns  # Returns all the columns that are categorical. (Categories..)

    def get_ord(self):
        return self.ordinals  # Returns all the columns that are ordinal (Less than or eql to 10 unique values).

    def get_num(self):
        return self.numeric_columns  # Returns all the columns that are numeric (ordinal + very numerical)

    def set_removed(self, removed_cols: list[str]):
        self.columns_to_ignore = removed_cols
        for col in removed_cols:
            try:
                if col in self.numeric_columns:
                    self.numeric_columns.remove(col)
                if col in self.very_numerical:
                    self.very_numerical.remove(col)
                if col in self.ordinals:
                    self.ordinals.remove(col)
                if col in self.categorical_columns:
                    self.categorical_columns.remove(col)
            except:
                print("Error in \'ColumnDivider::set_removed\'")

    def get_removed(self):
        return self.columns_to_ignore


def fill_null_model(data: pd.DataFrame, data_info: ColumnDivider):
    # Decided we are giving mean for very numerical, mode for ordinal (less than 10 features), and 'nan' new category for categorical.
    if data_info.get_removed():  # Not empty
        data.drop(data_info.get_removed(), axis=1, inplace=True, errors="ignore")  # Might already removed.

    categorical_columns = data_info.get_cat()
    ord_columns = data_info.get_ord()
    vnum_columns = data_info.get_vn()
    all_cols = data.columns
    for c in categorical_columns:
        if c not in all_cols:
            continue
        data[c] = pd.Categorical(data[c])
        if "nan" not in data[c].cat.categories:
            data[c] = data[c].cat.add_categories("nan")
        data[c] = data[c].fillna("nan")  # replacing na to the category 'na'.
        data[c] = data[c].apply(str)  # Added 25/02
    for c in ord_columns:
        if c not in all_cols:
            continue
        if data[c].mode().tolist():
            data[c] = data[c].fillna(data[c].mode()[0])
        else:
            data[c] = data[c].fillna(0)  # We can't really do anything else, except using 'get_rid_nullified'.

    for c in vnum_columns:
        if c not in all_cols:
            continue
        data[c] = data[c].fillna(data[c].mean(skipna=True))  # replacing na to mean of the values.

    return data


class MostCorrelated:
    most_correlated = {}
    binning = {}
    tf = ""
    di = None
    lists_len = 0
    data_info = None

    def __init__(self, data: pd.DataFrame, target_feature: str, num_correlated: int, data_info: ColumnDivider):
        # Remember! Todo - ignore 70% null columns.
        # data is a copy of the original train data, w/o null filling.
        self.tf = target_feature
        self.di = data_info
        if num_correlated >= len(data.columns) - 1:
            num_correlated = 10
        self.lists_len = num_correlated
        self.data_info = data_info
        # nullified_columns = getNullifiedColumns(data, 0.0)
        # data_info = ColumnDivider(data, target_feature, max_uniq)
        cat_cols, vnum_cols, num_cols, ord_cols = data_info.get_cat(), data_info.get_vn(), data_info.get_num(), data_info.get_ord()
        # 2. Fill with a new 'na' category:
        for c in cat_cols:
            data[c] = pd.Categorical(data[c])
            if "nan" not in data[c].cat.categories:
                data[c] = data[c].cat.add_categories("nan")
            data[c] = data[c].fillna("nan")

        for c in vnum_cols:
            try:
                data[c + '_binned'], self.binning[c] = pd.qcut(data[c], 5, labels=["very low", "low", "medium", "high",
                                                                                   "very high"], retbins=True)
            except:
                # sometimes for highly skewed data, we cannot perform qcut as most quantiles are equal
                data[c + '_binned'], self.binning[c] = pd.cut(data[c], 5,
                                                              labels=["very low", "low", "medium", "high", "very high"],
                                                              retbins=True)
            finally:
                self.binning[c] = np.concatenate(([-np.inf], self.binning[c][1:-1], [np.inf]))
                data[c + '_binned'] = pd.Categorical(data[c + '_binned'])
                if "nan" not in data[c + '_binned'].cat.categories:
                    data[c + '_binned'] = data[c + '_binned'].cat.add_categories("nan")
                data[c + '_binned'] = data[c + '_binned'].fillna("nan")


        # Binned very numericals! But didn't change v_numerical's nulls to appropriate.. now its np.nan

        for c in ord_cols:
            data[c + '_binned'] = pd.Categorical(data[c])
            if "nan" not in data[c + '_binned'].cat.categories:
                data[c + '_binned'] = data[c + '_binned'].cat.add_categories("nan")
            data[c + '_binned'] = data[c + '_binned'].fillna("nan")

        ord_cols_binned = [c + '_binned' for c in ord_cols]
        vnum_cols_binned = [c + '_binned' for c in vnum_cols]
        check_corr_list = list()
        check_corr_list.extend(ord_cols_binned)
        check_corr_list.extend(vnum_cols_binned)
        check_corr_list.extend(cat_cols)
        self.new_col_list = check_corr_list
        set_corr = {}
        for c_main in check_corr_list:
            list_p_vals = []
            for c_second in check_corr_list:
                if c_main != c_second:
                    contingency_table = pd.crosstab(data[c_main], data[c_second])
                    c, p, dof, expected = stats.chi2_contingency(contingency_table)
                    list_p_vals.append(p)
                else:
                    list_p_vals.append(1.0)
            idx = np.argpartition(np.array(list_p_vals), 10)  # Getting the 10 largest elements.
            set_corr[c_main] = idx[:10]

        for k, v in set_corr.items():
            self.most_correlated[k] = [check_corr_list[i] for i in v]
        # most_correlated = {k: [check_corr_list[i] for i in v] for k, v in set_corr.items()}

    def getMap(self) -> dict:
        return self.most_correlated

    def getMostCorrelated(self, feature: str) -> list:
        return self.most_correlated.get(feature)  # Returns null if it doesn't exists.

    def getBinnings(self) -> dict:
        return self.binning

    def getBinningColumns(self) -> list:
        return self.new_col_list

    def getTarget(self) -> str:
        return self.tf

    def processTest(self, data: pd.DataFrame, len_test: int, min_supp: float = 0.4, min_conf: float = 0.75):
        cat_cols, vnum_cols, num_cols, ord_cols = self.data_info.get_cat(), self.data_info.get_vn(), self.data_info.get_num(), self.data_info.get_ord()
        # 2. Fill with a new 'na' category:
        for c in cat_cols:
            data[c] = pd.Categorical(data[c])
            if "nan" not in data[c].cat.categories:  # Added 25/02
                data[c] = data[c].cat.add_categories("nan")  # That was still here
            data[c] = data[c].fillna("nan")
        for c in vnum_cols:
            data[c + '_binned'] = pd.cut(data[c], self.binning[c], labels=["very low", "low", "medium", "high",
                                                                           "very high"])  # This is probably the main giveaway for the error. TODO -fix.
            data[c + '_binned'] = pd.Categorical(data[c + '_binned'])
            if "nan" not in data[c + '_binned'].cat.categories:
                data[c + '_binned'] = data[c + '_binned'].cat.add_categories("nan")
            data[c + '_binned'] = data[c + '_binned'].fillna("nan")

        for c in ord_cols:
            data[c + '_binned'] = pd.Categorical(data[c])
            if "nan" not in data[c + '_binned'].cat.categories:
                data[c + '_binned'] = data[c + '_binned'].cat.add_categories("nan")
            data[c + '_binned'] = data[c + '_binned'].fillna("nan")

        rule_list = list()
        for c in self.new_col_list:  # All the new categories:
            records = data[self.most_correlated[c]].to_dict(orient='records')
            transactions = []
            for r in records:
                transactions.append(list(r.items()))
            itemsets, rules = ea.apriori(transactions, min_support=min_supp, min_confidence=min_conf,
                                         output_transaction_ids=False)
            rule_list.extend(rules)

        return rule_list, data


class Model:
    model_name: str = ""
    train_filename: str = ""
    this_model: Optional[xgb.XGBModel] = None  # The xgb regressor model.
    apriori_helper: Optional[MostCorrelated] = None  # MostCorrelated [change mc to self.apriori_helper].
    data_info: Optional[ColumnDivider] = None  # ColumnDivider [change cd to self.data_info].
    name_dict = set()  # Connects between the name the user gives to the test, and it's path.
    encoder: Optional[OneHotEncoder] = None
    err_margin = 0.05  # Make is so the user can enter it on it's own. todo

    def __init__(self, enc: Optional[OneHotEncoder] = None, data_info: Optional[ColumnDivider] = None):
        if enc is not None:
            print("enc is not none")
            self.encoder = enc
        if data_info is not None:
            print("data_info is not none")
            self.data_info = data_info

    def getModel(self) -> xgb.XGBRegressor:
        return self.this_model

    def createModel(self, name: str) -> bool:
        self.model_name = name
        self.tf = ""

        filename = get_file("Please upload your TRAIN file (.csv):\n", "csv")

        original_data = pd.read_csv(os.getcwd() + DirectorySlash + filename, low_memory=False)

        # Ask for a target feature, and check if there is such column:
        target_feature = None
        while target_feature is None:
            target_feature = input("\nPlease enter the target feature: ")
            if target_feature not in original_data.columns:
                target_feature = None

        self.tf = target_feature  # Save it into the model.

        # Consider what is the difference between Very numerical to Ordinal columns.
        max_uniq = None
        while max_uniq is None:
            max_uniq = input(
                "\nEnter maximum number of uniques allowed for ordinal (Will fill NAN values with mode instead of mean): ")
            max_uniq = int(max_uniq) if max_uniq.isdecimal() else None

        # Consider how many features we are going to try to find common between, when facing errors.
        best_correlated_features = None
        while best_correlated_features is None:
            best_correlated_features = input("\nEnter number of features we will test against, when facing errors: ")
            best_correlated_features = int(best_correlated_features) if best_correlated_features.isdecimal() else None

        data_for_binning = original_data.copy(deep=True)  # For binning.

        if self.data_info is None:
            self.data_info = ColumnDivider(data_for_binning, self.tf, max_uniq)
        self.apriori_helper = MostCorrelated(data_for_binning, self.tf, best_correlated_features,
                                             self.data_info)  # Will be needed for making apriori rules.

        categorical_columns = self.data_info.get_cat()

        data_for_ohe = original_data.copy(deep=True)

        data_for_ohe_filled = fill_null_model(data_for_ohe, self.data_info)  # Didnt need that??
        if self.encoder is None:
            self.encoder = OneHotEncoder(sparse=False)
            self.encoder.fit(data_for_ohe_filled[categorical_columns])
            data_for_ohe_filled_encoded = pd.DataFrame(self.encoder.transform(data_for_ohe_filled[categorical_columns]))
        else:
            data_for_ohe_filled_encoded = pd.DataFrame(self.encoder.transform(data_for_ohe_filled[categorical_columns]))

        data_for_ohe_filled_encoded.columns = self.encoder.get_feature_names_out(categorical_columns)

        data_for_ohe_filled.drop(categorical_columns, axis=1, inplace=True)

        OHE_data = pd.concat([data_for_ohe_filled, data_for_ohe_filled_encoded], axis=1)

        self.this_model = xgb.XGBRegressor(n_estimators=100, random_state=0)
        self.this_model.fit(OHE_data.drop(self.tf, axis=1), OHE_data[self.tf])

        # NOAM POST HERE
        test = pd.read_csv(os.getcwd() + DirectorySlash + 'final_test.csv')
        test = fill_null_model(test, self.data_info)
        test_categorical = pd.DataFrame(self.encoder.transform(test[categorical_columns]))
        test_categorical.columns = self.encoder.get_feature_names_out(categorical_columns)
        test = pd.concat([test_categorical, test.drop(categorical_columns, axis=1)], axis=1)
        our_feature_names = self.this_model.get_booster().feature_names

        our_model_test = test[our_feature_names]
        prediction_our_test = self.this_model.predict(our_model_test)

        try:
            with open(os.getcwd() + DirectorySlash + "results_final_test.csv.txt", "r") as f:
                results_final = [float(line.rstrip()) for line in f]
        except:
            print("Error: Can't open test results file.")

        value_errors = [abs(item[0] - item[1]) for item in zip(prediction_our_test, results_final)]
        mae = sum(value_errors) / len(value_errors)  # mean absolute error
        addValue(mae)
        # NOAM POST HERE

        return True

    def uploadTest(self) -> bool:

        filename = get_file("Please upload your TEST file (.csv):\n", "csv")

        test_data = pd.read_csv(os.getcwd() + DirectorySlash + filename)

        self.name_dict.add(filename)

        categorical_columns = self.data_info.get_cat()

        test_data_for_ohe_filled = fill_null_model(test_data, self.data_info)

        test_data_for_ohe_filled_encoded = pd.DataFrame(
            self.encoder.transform(test_data_for_ohe_filled[categorical_columns]))

        test_data_for_ohe_filled_encoded.columns = self.encoder.get_feature_names_out(categorical_columns)

        test_data_for_ohe_filled.drop(categorical_columns, axis=1, inplace=True)

        OHE_test_data = pd.concat([test_data_for_ohe_filled, test_data_for_ohe_filled_encoded], axis=1)

        predictions_test = self.this_model.predict(OHE_test_data)

        # Prediction should be $modelname.$testname.txt
        try:
            with open(os.getcwd() + DirectorySlash + self.model_name + '_' + filename + '.txt', 'w') as f:
                for idx, row_pred in enumerate(predictions_test):
                    if idx < len(predictions_test) - 1:  # If not the final prediction..
                        f.write(str(row_pred) + '\n')
                    else:  # If it is the final prediction.. (w/o newline)
                        f.write(str(row_pred))
        except:
            print("\nError. Couldn't write the prediction file (\'.txt\').\n")
            return False
        return True

    def uploadPrevResults(self) -> bool:
        print(
            "\nTEST RESULT file should be a text file, with each row having \nthe real target feature value of the compatible row in the data.\n")
        print("Make sure the file does NOT end with a newline.\n")

        test_result_filename = get_file("Please upload your TEST RESULTS file (.txt):\n", "txt")
        test_filename = 'nan'
        try:
            with open(os.getcwd() + DirectorySlash + test_result_filename, "r") as f:
                real_results = [line.rstrip() for line in f]

            while test_filename == 'nan' or test_filename not in self.name_dict:
                test_filename = input("Please enter a valid test filename:\n")

            with open(os.getcwd() + DirectorySlash + self.model_name + '_' + test_filename + '.txt', "r") as f:
                test_predictions = [line.rstrip() for line in f]
        except:
            print('Error. Failed to open saved prediction files.')

        if len(real_results) != len(test_predictions):
            print('Error. Not equal number of predictions and true values.\n')
            return False

        true_vs_pred = list()
        try:
            for true_val, pred in zip(real_results, test_predictions):
                true_vs_pred.append((float(true_val), float(pred)))
        except:
            print('Value Error. Couldn\'t transfer one of the true values or prediction values to numeric value.\n')
            return False

        # Continue to calculate errors (based on user first input on what is considered error), and do apriori and fit the model if found common itemsets~~


        err_margin_list = [(abs(item[0] - item[1]) / item[0]) for item in true_vs_pred]  # Why minus?
        err_indices = list()
        for idx, item in enumerate(err_margin_list):
            if item > self.err_margin:
                err_indices.append(idx)

        test_df = pd.read_csv(os.getcwd() + DirectorySlash + test_filename)
        test_df = fill_null_model(test_df, self.data_info)  # Added 25/02

        test_df = test_df.iloc[err_indices]


        sale_price_list = list()
        for i in err_indices:
            sale_price_list.append(true_vs_pred[i][0])

        test_df[self.tf] = sale_price_list  # We are inserting the true values of the rows (received from the user).

        test_rows_len = len(true_vs_pred)

        rule_list, test_transformed = self.apriori_helper.processTest(test_df.copy(deep=True), test_rows_len)

        rule_list = sorted(rule_list, key=lambda rule: (len(rule.lhs) + len(
            rule.rhs)) * rule.support)  # The biggest rule, which has high support (common itemset)

        row_atleast = min(math.ceil(0.5 * test_rows_len),
                          math.ceil(0.5 * len(
                              err_indices)))  # We take half of errors or test rows - whatever is lower.

        rules_atleast = math.ceil(0.5 * len(rule_list))

        indices_rows = set()

        if row_atleast != math.ceil(0.5 * len(err_indices)):
            for rule in rule_list:
                rules_atleast = rules_atleast - 1
                for index, row in test_transformed.iterrows():
                    if index in indices_rows:
                        continue
                    fail = False
                    for item in rule.rhs:
                        if row[item[0]] != item[1]:
                            fail = True
                            break
                    if fail == True:
                        continue
                    for item in rule.lhs:
                        if row[item[0]] != item[1]:
                            fail = True
                            break
                    if fail == False:
                        indices_rows.add(index)
                if rules_atleast <= 0 or len(indices_rows) >= row_atleast:
                    break
            indices_rows = list(indices_rows)
            # NOAM PRINT
        else:
            indices_rows = []


        if indices_rows:
            insert_these = []
            for index, row in test_df.iterrows():
                if index in indices_rows:
                    insert_these.append(row.values)
            # take the rows from test_df that has the same indices as transformed (we took indices of rows that
            # comply with our rules after all..)
            test_df = pd.DataFrame(insert_these, columns=test_df.columns).reset_index()
        else:
            pass
            # insert_these = []
            # for index, row in test_df.iterrows():
            #     insert_these.append(row.values)
            # test_df = pd.DataFrame(insert_these, columns=test_df.columns).reset_index()

        if test_df.empty:  # No errors
            print("There were no errors!")
            return True

        categorical_columns = self.data_info.get_cat()

        test_data_fit_filled = fill_null_model(test_df, self.data_info)

        test_data_fit_filled_encoded = pd.DataFrame(self.encoder.transform(test_data_fit_filled[categorical_columns]))

        test_data_fit_filled_encoded.columns = self.encoder.get_feature_names_out(categorical_columns)

        test_data_fit_filled.drop(categorical_columns, axis=1, inplace=True)

        OHE_test_data_fit = pd.concat(
            [test_data_fit_filled.reset_index(drop=True), test_data_fit_filled_encoded.reset_index(drop=True)],
            axis=1)
        if 'index' in OHE_test_data_fit:
            OHE_test_data_fit.drop('index', axis=1, inplace=True)

        self.this_model = self.this_model.fit(OHE_test_data_fit.drop(self.tf, axis=1), OHE_test_data_fit[self.tf],
                                              xgb_model=self.this_model.get_booster())

        test = pd.read_csv(os.getcwd() + DirectorySlash + 'final_test.csv')
        test = fill_null_model(test, self.data_info)
        test_categorical = pd.DataFrame(self.encoder.transform(test[categorical_columns]))
        test_categorical.columns = self.encoder.get_feature_names_out(categorical_columns)
        test = pd.concat([test_categorical, test.drop(categorical_columns, axis=1)], axis=1)
        our_feature_names = self.this_model.get_booster().feature_names

        our_model_test = test[our_feature_names]
        prediction_our_test = self.this_model.predict(our_model_test)

        try:
            with open(os.getcwd() + DirectorySlash + "results_final_test.csv.txt", "r") as f:
                results_final = [float(line.rstrip()) for line in f]
        except:
            print("Error: Can't open test results file.")

        value_errors = [abs(item[0] - item[1]) for item in zip(prediction_our_test, results_final)]
        mae = sum(value_errors) / len(value_errors)  # mean absolute error
        addValue(mae)

        return True


def divide_data(target_feature: str, path: str = os.getcwd() + DirectorySlash + 'train.csv',
                name_tuple: tuple[str] = ("test1", "test2", "test3"),
                each_frac: float = 0.2, train_frac: float = 0.2, train_name: str = "final_train.csv",
                test_name: str = "final_test.csv") -> \
        tuple[Optional[OneHotEncoder], Optional[ColumnDivider], bool]:
    if each_frac > 1.0 or len(name_tuple) * each_frac > 1.0:  # We should leave some stuff for the final test.
        return None, None, False

    data: pd.DataFrame = pd.read_csv(path, low_memory=False)

    data_info = ColumnDivider(data, target_feature, 10)

    data: pd.DataFrame = fill_null_model(data,
                                         data_info)  # Need to fill null by mean and mode of the 'big' train file.

    categorical_columns = data_info.get_cat()

    row_len = len(data.index)
    cols_removed: list[str] = list()
    max_unique_categories = min(row_len, 100)
    for c in categorical_columns:
        if data[c].nunique() >= max_unique_categories:
            data.drop(c, axis=1, inplace=True)
            cols_removed.append(c)
            continue

        data[c] = data[c].apply(str)

    data_info.set_removed(cols_removed)



    saved_train = data.sample(frac=train_frac, random_state=200)
    test = data.drop(saved_train.index)

    samples: list[Optional[pd.DataFrame]] = [None] * len(name_tuple)
    for i in range(len(name_tuple)):
        samples[i] = test.sample(frac=each_frac, random_state=150)

    answers: list[Optional[pd.DataFrame]] = [None] * len(samples)
    for i in range(len(samples)):
        answers[i] = samples[i][target_feature]
        try:
            final_prediction_idx = len(answers[i]) - 1
            with open(os.getcwd() + DirectorySlash + 'results_' + name_tuple[i] + '.txt', 'w') as f:
                for idx, ans in enumerate(answers[i]):
                    if idx < final_prediction_idx:  # If not the final prediction..
                        f.write(str(ans) + '\n')
                    else:  # If it is the final prediction.. (w/o newline)
                        f.write(str(ans))
        except:
            print("Error in saving answers.")

    for i in range(len(samples)):
        samples[i].drop(target_feature, axis=1, inplace=True)

    for i in range(len(samples)):
        samples[i].to_csv(os.getcwd() + DirectorySlash + name_tuple[i] + '.csv', index=False)
        test.drop(samples[i].index, inplace=True, errors='ignore')

    # remember to save test's answers to a text file as we used so far, and drop it from the dataframe.
    test_answers: list = test[target_feature].values.tolist()

    # Writing the answers to a file.
    try:
        final_prediction_idx = len(test_answers) - 1
        with open(os.getcwd() + DirectorySlash + 'results_' + test_name + '.txt', 'w') as f:
            for idx, ans in enumerate(test_answers):
                if idx < final_prediction_idx:  # If not the final prediction..
                    f.write(str(ans) + '\n')
                else:  # If it is the final prediction.. (w/o newline)
                    f.write(str(ans))
    except:
        print("Error in saving answers.")

    test.drop(target_feature, axis=1, inplace=True, errors='ignore')
    test.to_csv(os.getcwd() + DirectorySlash + test_name, index=False)
    saved_train.to_csv(os.getcwd() + DirectorySlash + train_name, index=False)

    encoder = OneHotEncoder(sparse=False)

    encoder.fit(data[categorical_columns])

    try:
        with open(os.getcwd() + DirectorySlash + "encoder", "wb") as f:
            pickle.dump(encoder, f)

        with open(os.getcwd() + DirectorySlash + "DataInfo", "wb") as f:
            pickle.dump(data_info, f)
    except:
        return encoder, data_info, False

    return encoder, data_info, True


def our_project(enc: Optional[OneHotEncoder] = None, data_info: Optional[ColumnDivider] = None):
    models = {}  # Todo use that to contain all Models and their name
    while True:
        option = 0
        while option < 1 or option > 4:
            option = input(
                "Please choose:\n1. Create New Model.\n2. Upload another test file for prediction.\n3. Upload results for a previous test file.\n4. Return a requested model.\nEnter option: ")
            option = int(option) if option.isdecimal() else 0
        if option == 1:
            model_name = "nan"
            while model_name == "nan":
                model_name = input("\nEnter a name for the new model (except \'nan\'): \n")
            models[model_name] = Model(enc, data_info)
            if not models[model_name].createModel(model_name):
                del models[model_name]

        elif option == 2:
            model_name = "nan"
            while model_name == "nan" or model_name not in models.keys():
                model_name = input("\nEnter the name of the model (except \'nan\'): \n")
            models[model_name].uploadTest()  # We dont care if it is failed or not.

        elif option == 3:
            model_name = "nan"
            while model_name == "nan" or model_name not in models.keys():
                model_name = input("\nEnter the name of the model (except \'nan\'): \n")
            # models[model_name].uploadPrevResults()
            models[model_name].uploadPrevResults()
        else:
            model_name = "nan"
            while model_name == "nan" or model_name not in models.keys():
                model_name = input("\nEnter the name of the model (except \'nan\'): \n")
            showOnGraph()
            return models[model_name].getModel()


# def main_house_price():  # Our main shows us the difference between our solution and a normal training (training on all mistakes)
#     data = pd.read_csv(os.getcwd() + DirectorySlash + 'train.csv')
#
#     # train = data.sample(frac=0.5,random_state=200)
#     # test = df.drop(train.index)
#
#     # test1, test2, test3 = test.sample(frac=0.25 ,random_state=190), test.sample(frac=0.25 ,random_state=180), test.sample(frac=0.25 ,random_state=170)
#
#     # test1_answer, test2_answer, test3_answer = test1['SalePrice'], test2['SalePrice'], test3['SalePrice']
#     # test1, test2, test3 = test1.drop('SalePrice', axis=1), test2.drop('SalePrice', axis=1), test3.drop('SalePrice', axis=1)
#
#     # So far need to - 1) make the normal model, run the train fit, run the tests, check based on the answers the wrong predictions (+5%) and FIT ALL these lines into the model. run the whole test and check for grade.
#     # Then, make the answers as TXT files,make the tests as CSV files, and make the train CSV file.
#
#     data_info = ColumnDivider(data.copy(deep=True), 'SalePrice', 10)
#     categorical_columns = data_info.get_cat()
#
#     saved_train = data.sample(frac=0.05, random_state=200)
#     test = data.drop(saved_train.index)
#     saved_test1, saved_test2, saved_test3 = test.sample(frac=0.2, random_state=190), test.sample(frac=0.2,
#                                                                                                  random_state=180), test.sample(
#         frac=0.2, random_state=170)
#
#     saved_test1_answers, saved_test2_answers, saved_test3_answers = saved_test1['SalePrice'], saved_test2['SalePrice'], \
#         saved_test3['SalePrice']
#     saved_test1, saved_test2, saved_test3 = saved_test1.drop('SalePrice', axis=1), saved_test2.drop('SalePrice',
#                                                                                                     axis=1), saved_test3.drop(
#         'SalePrice', axis=1)
#
#     data_filled = fill_null_model(data.copy(deep=True), data_info)
#
#     encoder = OneHotEncoder(sparse=False)
#
#     data_filled_encoded = pd.DataFrame(encoder.fit_transform(data_filled[categorical_columns]))
#
#     data_filled_encoded.columns = encoder.get_feature_names_out(categorical_columns)
#
#     data_filled.drop(categorical_columns, axis=1, inplace=True)
#
#     OHE_data = pd.concat([data_filled.reset_index(drop=True), data_filled_encoded.reset_index(drop=True)], axis=1)
#
#     train = OHE_data.sample(frac=0.5, random_state=200)
#     test = OHE_data.drop(train.index)
#
#     test1, test2, test3 = test.sample(frac=0.2, random_state=190), test.sample(frac=0.2, random_state=180), test.sample(
#         frac=0.2, random_state=170)
#
#     test1_answers, test2_answers, test3_answers = test1['SalePrice'], test2['SalePrice'], test3['SalePrice']
#     test_final = test.copy(deep=True)
#     test_final.drop(test1.index, inplace=True, errors='ignore')
#     test_final.drop(test2.index, inplace=True, errors='ignore')
#     test_final.drop(test3.index, inplace=True, errors='ignore')
#     test1, test2, test3 = test1.drop('SalePrice', axis=1), test2.drop('SalePrice', axis=1), test3.drop('SalePrice',
#                                                                                                        axis=1)
#
#     normal_model = xgb.XGBRegressor(n_estimators=100, random_state=0)
#
#     normal_model.fit(train.drop('SalePrice', axis=1), train['SalePrice'])
#
#     # Test1
#     prediction_normal_test1 = normal_model.predict(test1)
#
#     test1_answers_list = test1_answers.values.tolist()
#
#     indices_list = list()
#     for i in range(len(test1_answers)):
#         if abs(prediction_normal_test1[i] - test1_answers_list[i]) / test1_answers_list[i] < 0.05:
#             indices_list.append(i)
#
#     normal_model = normal_model.fit(test1.iloc[indices_list], test1_answers.iloc[indices_list],
#                                     xgb_model=normal_model.get_booster())
#
#     # Test2
#     prediction_normal_test2 = normal_model.predict(test2)
#     test2_answers_list = test2_answers.values.tolist()
#
#     indices_list = list()
#     for i in range(len(test2_answers)):
#         if abs(prediction_normal_test2[i] - test2_answers_list[i]) / test2_answers_list[i] < 0.05:
#             indices_list.append(i)
#
#     normal_model = normal_model.fit(test2.iloc[indices_list], test2_answers.iloc[indices_list],
#                                     xgb_model=normal_model.get_booster())
#
#     # Test3
#     prediction_normal_test3 = normal_model.predict(test3)
#     test3_answers_list = test3_answers.values.tolist()
#
#     indices_list = list()
#     for i in range(len(test3_answers)):
#         if abs(prediction_normal_test3[i] - test3_answers_list[i]) / test3_answers_list[i] < 0.05:
#             indices_list.append(i)
#
#     normal_model = normal_model.fit(test3.iloc[indices_list], test3_answers.iloc[indices_list],
#                                     xgb_model=normal_model.get_booster())
#
#     final_predictions_normal = normal_model.predict(test_final.drop('SalePrice', axis=1))  # normal model.
#     true_values_test = test_final['SalePrice'].values.tolist()
#
#     count_failed = 0
#     for i in range(len(true_values_test)):
#         if abs(final_predictions_normal[i] - true_values_test[i]) / true_values_test[i] < 0.05:
#             count_failed = count_failed + 1
#
#     normal_success = count_failed / len(true_values_test)
#
#     train = data.sample(frac=0.5, random_state=200)
#     test = data.drop(train.index)
#     test1, test2, test3 = test.sample(frac=0.2, random_state=190), test.sample(frac=0.2, random_state=180), test.sample(
#         frac=0.2, random_state=170)
#
#     test1_answers, test2_answers, test3_answers = test1['SalePrice'].values.tolist(), test2[
#         'SalePrice'].values.tolist(), test3['SalePrice'].values.tolist()
#     test1, test2, test3 = test1.drop('SalePrice', axis=1), test2.drop('SalePrice', axis=1), test3.drop('SalePrice',
#                                                                                                        axis=1)
#
#     saved_train.to_csv(Path(os.getcwd() + DirectorySlash + 'train1.csv'), index=False)
#
#     saved_test1.to_csv(Path(os.getcwd() + DirectorySlash + 'test1.csv'), index=False)
#     saved_test2.to_csv(Path(os.getcwd() + DirectorySlash + 'test2.csv'), index=False)
#     saved_test3.to_csv(Path(os.getcwd() + DirectorySlash + 'test3.csv'), index=False)
#
#     try:
#         with open(os.getcwd() + DirectorySlash + 'results_test1.txt', 'w') as f:
#             for idx, ans in enumerate(saved_test1_answers):
#                 if idx < len(saved_test1_answers) - 1:  # If not the final prediction..
#                     f.write(str(ans) + '\n')
#                 else:  # If it is the final prediction.. (w/o newline)
#                     f.write(str(ans))
#     except:
#         print("\nError. Couldn't write the prediction file (\'.txt\').\n")
#
#     try:
#         with open(os.getcwd() + DirectorySlash + 'results_test2.txt', 'w') as f:
#             for idx, ans in enumerate(saved_test2_answers):
#                 if idx < len(saved_test2_answers) - 1:  # If not the final prediction..
#                     f.write(str(ans) + '\n')
#                 else:  # If it is the final prediction.. (w/o newline)
#                     f.write(str(ans))
#     except:
#         print("\nError. Couldn't write the prediction file (\'.txt\').\n")
#
#     try:
#         with open(os.getcwd() + DirectorySlash + 'results_test3.txt', 'w') as f:
#             for idx, ans in enumerate(saved_test3_answers):
#                 if idx < len(saved_test3_answers) - 1:  # If not the final prediction..
#                     f.write(str(ans) + '\n')
#                 else:  # If it is the final prediction.. (w/o newline)
#                     f.write(str(ans))
#     except:
#         print("\nError. Couldn't write the prediction file (\'.txt\').\n")
#
#     our_model = our_project(encoder, data_info)
#     our_predictions = our_model.predict(test_final.drop('SalePrice', axis=1))
#     print('our_pred\n', our_predictions)
#     count_failed = 0
#     for i in range(len(true_values_test)):
#         if abs(our_predictions[i] - true_values_test[i]) / true_values_test[i] < 0.05:
#             count_failed = count_failed + 1
#
#     our_success = count_failed / len(true_values_test)
#
#     print('Our success: ', our_success, '\nNormal Success: ', normal_success)


def general_main():
    # 1) Remove the comment from all the lines below, continue and make main func. 2) Do another dataset!!!
    target_feature = input("Please enter the dataset's target feature:\n")
    primary_dataset_file = input("Please enter the dataset's filename (with the .csv extension) :\n")
    # target_feature = 'price'
    result_tuple: tuple[Optional[OneHotEncoder], Optional[ColumnDivider], bool] = divide_data(target_feature,
                                                                                              path=os.getcwd() + DirectorySlash + primary_dataset_file,
                                                                                              train_frac=0.05)

    encoder: Optional[OneHotEncoder] = result_tuple[0]
    data_info: Optional[ColumnDivider] = result_tuple[1]

    is_saved: bool = result_tuple[2]
    #

    train = pd.read_csv(os.getcwd() + DirectorySlash + 'final_train.csv')
    test = pd.read_csv(os.getcwd() + DirectorySlash + 'final_test.csv')
    test1 = pd.read_csv(os.getcwd() + DirectorySlash + 'test1.csv')
    test2 = pd.read_csv(os.getcwd() + DirectorySlash + 'test2.csv')
    test3 = pd.read_csv(os.getcwd() + DirectorySlash + 'test3.csv')

    categorical_columns = data_info.get_cat()

    # DO ENCODER!!!
    train = fill_null_model(train, data_info)
    train_categorical = pd.DataFrame(encoder.transform(train[categorical_columns]))
    train_categorical.columns = encoder.get_feature_names_out(categorical_columns)
    train = pd.concat([train_categorical, train.drop(categorical_columns, axis=1)], axis=1)

    # Final test
    test = fill_null_model(test, data_info)
    test_categorical = pd.DataFrame(encoder.transform(test[categorical_columns]))
    test_categorical.columns = encoder.get_feature_names_out(categorical_columns)
    test = pd.concat([test_categorical, test.drop(categorical_columns, axis=1)], axis=1)

    try:
        with open(os.getcwd() + DirectorySlash + "results_test1.txt", "r") as f:
            results1 = [float(line.rstrip()) for line in f]
        with open(os.getcwd() + DirectorySlash + "results_test2.txt", "r") as f:
            results2 = [float(line.rstrip()) for line in f]
        with open(os.getcwd() + DirectorySlash + "results_test3.txt", "r") as f:
            results3 = [float(line.rstrip()) for line in f]
        with open(os.getcwd() + DirectorySlash + "results_final_test.csv.txt", "r") as f:
            results_final = [float(line.rstrip()) for line in f]
    except:
        print("Error: Can't open test results file.")

    # Our model
    our_model: xgb.XGBModel = our_project(encoder, data_info)
    our_feature_names = our_model.get_booster().feature_names

    our_model_test = test[our_feature_names]
    prediction_our_test = our_model.predict(our_model_test)

    # Next step - save in each test iteration (prevTestResult) the lines where we found they have features in common (the rows we fit the model on) AND how close they are to the target (after learning).

    # Normal model
    normal_model = xgb.XGBRegressor(n_estimators=100, random_state=0)
    normal_model.fit(train.drop(target_feature, axis=1), train[target_feature])

    normal_feature_names = normal_model.get_booster().feature_names
    normal_model_test = test[normal_feature_names]
    prediction_normal_test = normal_model.predict(normal_model_test)
    value_errors = [abs(item[0] - item[1]) for item in zip(prediction_normal_test, results_final)]
    mae = sum(value_errors) / len(value_errors)  # mean absolute error
    addValue(mae)



    # Test1
    test1 = fill_null_model(test1, data_info)
    test1_categorical = pd.DataFrame(encoder.transform(test1[categorical_columns]))
    test1_categorical.columns = encoder.get_feature_names_out(categorical_columns)
    test1 = pd.concat([test1_categorical, test1.drop(categorical_columns, axis=1)], axis=1)

    prediction_normal_test1 = normal_model.predict(test1)


    indices_list = list()
    for i in range(len(results1)):
        if abs(prediction_normal_test1[i] - results1[i]) / results1[i] > 0.05:
            indices_list.append(i)

    test1[target_feature] = results1
    test1 = test1.iloc[indices_list]
    normal_model = normal_model.fit(test1.drop(target_feature, axis=1), test1[target_feature],
                                    xgb_model=normal_model.get_booster())

    normal_feature_names = normal_model.get_booster().feature_names
    normal_model_test = test[normal_feature_names]
    prediction_normal_test = normal_model.predict(normal_model_test)
    value_errors = [abs(item[0] - item[1]) for item in zip(prediction_normal_test, results_final)]
    mae = sum(value_errors) / len(value_errors)  # mean absolute error
    addValue(mae)

    # Test2
    test2 = fill_null_model(test2, data_info)
    test2_categorical = pd.DataFrame(encoder.transform(test2[categorical_columns]))
    test2_categorical.columns = encoder.get_feature_names_out(categorical_columns)
    test2 = pd.concat([test2_categorical, test2.drop(categorical_columns, axis=1)], axis=1)

    prediction_normal_test2 = normal_model.predict(test2)

    indices_list = list()
    for i in range(len(results2)):
        if abs(prediction_normal_test2[i] - results2[i]) / results2[i] < 0.05:
            indices_list.append(i)



    test2[target_feature] = results2
    test2 = test2.iloc[indices_list]
    normal_model = normal_model.fit(test2.drop(target_feature, axis=1), test2[target_feature],
                                    xgb_model=normal_model.get_booster())

    normal_feature_names = normal_model.get_booster().feature_names
    normal_model_test = test[normal_feature_names]
    prediction_normal_test = normal_model.predict(normal_model_test)
    value_errors = [abs(item[0] - item[1]) for item in zip(prediction_normal_test, results_final)]
    mae = sum(value_errors) / len(value_errors)  # mean absolute error
    addValue(mae)

    # Test3
    test3 = fill_null_model(test3, data_info)
    test3_categorical = pd.DataFrame(encoder.transform(test3[categorical_columns]))
    test3_categorical.columns = encoder.get_feature_names_out(categorical_columns)
    test3 = pd.concat([test3_categorical, test3.drop(categorical_columns, axis=1)], axis=1)

    prediction_normal_test3 = normal_model.predict(test3)

    indices_list = list()
    for i in range(len(results3)):
        if abs(prediction_normal_test3[i] - results3[i]) / results3[i] < 0.05:
            indices_list.append(i)

    test3[target_feature] = results3
    test3 = test3.iloc[indices_list]
    normal_model = normal_model.fit(test3.drop(target_feature, axis=1), test3[target_feature],
                                    xgb_model=normal_model.get_booster())

    normal_feature_names = normal_model.get_booster().feature_names
    normal_model_test = test[normal_feature_names]
    prediction_normal_test = normal_model.predict(normal_model_test)
    value_errors = [abs(item[0] - item[1]) for item in zip(prediction_normal_test, results_final)]
    mae = sum(value_errors) / len(value_errors)  # mean absolute error
    addValue(mae)



    normal_feature_names = normal_model.get_booster().feature_names

    normal_model_test = test[normal_feature_names]

    prediction_normal_test = normal_model.predict(normal_model_test)

    print(prediction_our_test)
    print(prediction_normal_test)
    showOnGraph()


if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    general_main()
