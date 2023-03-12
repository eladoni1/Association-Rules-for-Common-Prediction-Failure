# Association-Rules-for-Common-Prediction-Failure
Article for TDS course by Dr. Amit Somech at BIU


The dataset and requirements.txt files are placed in the same directory as the script. The necessary libraries are installed by running the "pip install -r requirements.txt" command, which installs all the libraries listed in the txt file.
Next, the "sudo apt-get install python3-tk" command is used to install the directory for selecting files by the console. Once this is done, any dataset can be placed next to the script and run.

The dataset file is supposed to be in the same directory as the script.

Running the script, the general function will run, which will require us to specify the "target feature" inside the dataset csv file, and the filename itself containing the data.

The data is then separated, and divided into different files (Train dataset, and tests dataset, and a validation dataset).

We will then choose to create a model (by pressing 1 on the option menu).
The model will require to be named, choose a train dataset file (.csv), and choose the parameters required - target feature, number of unqiues to have a feature to be considered ordinal (Default value: 10), and how large will a rule get (Default value: 10).

The data will then be trained upon the model.

Choosing the option to upload a test file (pressing 2 on the option menu) -
Select one of the test files in the script directory (or custom made test file), and let the model return it's predictions, which will then be saved into an appropriate file (.txt), inside the directory of the script. 

Choosing the option to upload a results file (by pressing 3 on the option menu) - 
Select one of the results file in the script directory (or custom made results file), and let the model compare it's previous predictions to the real value of the appropriate test's file rows, and then calculate the rules derived from the model's prediction faults (currently set to 5% deviation from the actual value), filter the faulty rows by these rules, and eventually train them upon the model.

Choosing the option to return the model - returning the model (object by the xgboost library) after the function is finished.

We will then be shown the success of our model via graphs (Mean Absolute Error).



