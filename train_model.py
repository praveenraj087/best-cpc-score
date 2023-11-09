


import sys
import os, numpy as np, scipy as sp, scipy.io
import mne
from sklearn.impute import SimpleImputer
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error



class LinearClassifierOutcome(nn.Module):
    def __init__(self, input_size):
        super(LinearClassifierOutcome, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.classifier(x)
        return x
    
    def predict_proba(self,x):
        proba = self.forward(x)
        return proba[:, 0].item()
    
class LinearClassifierCPC(nn.Module):
    def __init__(self, input_size):
        super(LinearClassifierCPC, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


import os, numpy as np, scipy as sp, scipy.io


# Find the folders with data files.
def find_data_folders(root_folder):
    data_folders = list()
    for x in os.listdir(root_folder):
        data_folder = os.path.join(root_folder, x)
        if os.path.isdir(data_folder):
            data_folders.append(x)
    return sorted(data_folders)

def load_data(data_folder, patient_id):
    # Define file location.
    patient_metadata_file = os.path.join(data_folder, patient_id, patient_id + '.txt')
    recording_metadata_file = os.path.join(data_folder, patient_id, patient_id + '.tsv')

    # Load non-recording data.
    patient_metadata = load_text_file(patient_metadata_file)
    recording_metadata = load_text_file(recording_metadata_file)

    # Load recordings.
    recordings = list()
    recording_ids = get_recording_ids(recording_metadata)
    for recording_id in recording_ids:
        if not is_nan(recording_id):
            recording_location = os.path.join(data_folder, patient_id, recording_id)
            recording_data, sampling_frequency, channels = load_recording(recording_location)
            # print("Load Recording")
        else:
            recording_data = None
            sampling_frequency = None
            channels = None
        recordings.append((recording_data, sampling_frequency, channels))

    return patient_metadata, recording_metadata, recordings

def load_clinic_data(data_folder, patient_id):
    # Define file location.
    patient_metadata_file = os.path.join(data_folder, patient_id, patient_id + '.txt')

    # Load non-recording data.
    patient_metadata = load_text_file(patient_metadata_file)

    return patient_metadata

# Load the WFDB data (but not all possible WFDB files).
def load_recording(record_name):
    # Allow either the record name or the header filename.
    root, ext = os.path.splitext(record_name)
    if ext=='':
        header_file = record_name + '.hea'
    else:
        header_file = record_name

    # Load the header file.
    if not os.path.isfile(header_file):
        raise FileNotFoundError('{} recording not found.'.format(record_name))

    with open(header_file, 'r') as f:
        header = [l.strip() for l in f.readlines() if l.strip()]
    # print("Header" , header)
    # Parse the header file.
    record_name = None
    num_signals = None
    sampling_frequency = None
    num_samples = None
    signal_files = list()
    gains = list()
    offsets = list()
    channels = list()
    initial_values = list()
    checksums = list()

    for i, l in enumerate(header):
        arrs = [arr.strip() for arr in l.split(' ')]
        # Parse the record line.
        if i==0:
            record_name = arrs[0]
            num_signals = int(arrs[1])
            sampling_frequency = float(arrs[2])
            num_samples = int(arrs[3])
        # Parse the signal specification lines.
        else:
            signal_file = arrs[0]
            gain = float(arrs[2].split('/')[0])
            offset = int(arrs[4])
            initial_value = int(arrs[5])
            checksum = int(arrs[6])
            channel = arrs[8]
            signal_files.append(signal_file)
            gains.append(gain)
            offsets.append(offset)
            initial_values.append(initial_value)
            checksums.append(checksum)
            channels.append(channel)

    # Check that the header file only references one signal file. WFDB format  allows for multiple signal files, but we have not
    # implemented that here for simplicity.
    num_signal_files = len(set(signal_files))
    if num_signal_files!=1:
        raise NotImplementedError('The header file {}'.format(header_file) \
            + ' references {} signal files; one signal file expected.'.format(num_signal_files))

    # Load the signal file.
    head, tail = os.path.split(header_file)
    signal_file = os.path.join(head, list(signal_files)[0])
    data = np.asarray(sp.io.loadmat(signal_file)['val'])

    # Check that the dimensions of the signal data in the signal file is consistent with the dimensions for the signal data given
    # in the header file.
    num_channels = len(channels)
    if np.shape(data)!=(num_channels, num_samples):
        raise ValueError('The header file {}'.format(header_file) \
            + ' is inconsistent with the dimensions of the signal file.')

    # Check that the initial value and checksums for the signal data in the signal file are consistent with the initial value and
    # checksums for the signal data given in the header file.
    for i in range(num_channels):
        if data[i, 0]!=initial_values[i]:
            raise ValueError('The initial value in header file {}'.format(header_file) \
                + ' is inconsistent with the initial value for channel'.format(channels[i]))
        if np.sum(data[i, :])!=checksums[i]:
            raise ValueError('The checksum in header file {}'.format(header_file) \
                + ' is inconsistent with the initial value for channel'.format(channels[i]))

    # Rescale the signal data using the ADC gains and ADC offsets.
    rescaled_data = np.zeros(np.shape(data), dtype=np.float32)
    for i in range(num_channels):
        rescaled_data[i, :] = (data[i, :]-offsets[i])/gains[i]

    return rescaled_data, sampling_frequency, channels

# Reorder/reselect the channels.
def reorder_recording_channels(current_data, current_channels, reordered_channels):
    if current_channels == reordered_channels:
        return current_data
    else:
        indices = list()
        for channel in reordered_channels:
            if channel in current_channels:
                i = current_channels.index(channel)
                indices.append(i)
        num_channels = len(reordered_channels)
        num_samples = np.shape(current_data)[1]
        reordered_data = np.zeros((num_channels, num_samples))
        reordered_data[:, :] = current_data[indices, :]
        return reordered_data

### Helper data I/O functions

# Load text file as a string.
def load_text_file(filename):
    with open(filename, 'r') as f:
        data = f.read()
    return data

# Parse a value.
def cast_variable(variable, variable_type, preserve_nan=True):
    if preserve_nan and is_nan(variable):
        variable = float('nan')
    else:
        if variable_type == bool:
            variable = sanitize_boolean_value(variable)
        elif variable_type == int:
            variable = sanitize_integer_value(variable)
        elif variable_type == float:
            variable = sanitize_scalar_value(variable)
        else:
            variable = variable_type(variable)
    return variable

# Get a variable from the patient metadata.
def get_variable(text, variable_name, variable_type):
    variable = None
    for l in text.split('\n'):
        if l.startswith(variable_name):
            variable = l.split(':')[1].strip()
            variable = cast_variable(variable, variable_type)
            return variable

# Get a column from the recording metadata.
def get_column(string, column, variable_type, sep='\t'):
    variables = list()
    for i, l in enumerate(string.split('\n')):
        arrs = [arr.strip() for arr in l.split(sep) if arr.strip()]
        if i==0:
            column_index = arrs.index(column)
        elif arrs:
            variable = arrs[column_index]
            variable = cast_variable(variable, variable_type)
            variables.append(variable)
    return np.asarray(variables)


# Get the patient ID variable from the patient data.
def get_patient_id(string):
    return get_variable(string, 'Patient', str)

# Get the age variable (in years) from the patient data.
def get_age(string):
    return get_variable(string, 'Age', int)
def get_cpc(string):
    return get_variable(string, 'CPC', int)
# Get the sex variable from the patient data.
def get_sex(string):
    return get_variable(string, 'Sex', str)

# Get the ROSC variable (in minutes) from the patient data.
def get_rosc(string):
    return get_variable(string, 'ROSC', int)

# Get the OHCA variable from the patient data.
def get_ohca(string):
    return get_variable(string, 'OHCA', bool)

# Get the VFib variable from the patient data.
def get_vfib(string):
    return get_variable(string, 'VFib', bool)

# Get the TTM variable (in Celsius) from the patient data.
def get_ttm(string):
    return get_variable(string, 'TTM', int)

# Get the Outcome variable from the patient data.
def get_outcome(string):
    variable = get_variable(string, 'Outcome', str)
    if variable is None or is_nan(variable):
        raise ValueError('No outcome available. Is your code trying to load labels from the hidden data?')
    if variable == 'Good':
        variable = 0
    elif variable == 'Poor':
        variable = 1
    return variable

# Get the Outcome probability variable from the patient data.
def get_outcome_probability(string):
    variable = sanitize_scalar_value(get_variable(string, 'Outcome probability', str))
    if variable is None or is_nan(variable):
        raise ValueError('No outcome available. Is your code trying to load labels from the hidden data?')
    return variable

# Get the CPC variable from the patient data.
def get_cpc(string):
    variable = sanitize_scalar_value(get_variable(string, 'CPC', str))
    if variable is None or is_nan(variable):
        raise ValueError('No CPC score available. Is your code trying to load labels from the hidden data?')
    return variable

# Get the hour number column from the patient data.
def get_hours(string):
    return get_column(string, 'Hour', int)

# Get the time column from the patient data.
def get_times(string):
    return get_column(string, 'Time', str)

# Get the quality score column from the patient data.
def get_quality_scores(string):
    return get_column(string, 'Quality', float)

# Get the recording IDs column from the patient data.
def get_recording_ids(string):
    return get_column(string, 'Record', str)

### Label and output I/O functions

# Load the labels for one file.
def load_label(string):
    if os.path.isfile(string):
        string = load_text_file(string)

    outcome = get_outcome(string)
    cpc = get_cpc(string)

    return outcome, cpc

# Load all the labels for all of the files in a folder.
def load_labels(folder):
    patient_folders = find_data_folders(folder)
    num_patients = len(patient_folders)

    patient_ids = list()
    outcomes = np.zeros(num_patients, dtype=np.bool_)
    cpcs = np.zeros(num_patients, dtype=np.float64)

    for i in range(num_patients):
        patient_metadata_file = os.path.join(folder, patient_folders[i], patient_folders[i] + '.txt')
        patient_metadata = load_text_file(patient_metadata_file)

        patient_ids.append(get_patient_id(patient_metadata))
        outcomes[i] = get_outcome(patient_metadata)
        cpcs[i] = get_cpc(patient_metadata)

    return patient_ids, outcomes, cpcs


# Check if a variable is a number or represents a number.
def is_number(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False

# Check if a variable is an integer or represents an integer.
def is_integer(x):
    if is_number(x):
        return float(x).is_integer()
    else:
        return False

# Checking if a variable is a boolean
def is_boolean(x):
    if (is_number(x) and float(x)==0) or (remove_extra_characters(x) in ('False', 'false', 'FALSE', 'F', 'f')):
        return True
    elif (is_number(x) and float(x)==1) or (remove_extra_characters(x) in ('True', 'true', 'TRUE', 'T', 't')):
        return True
    else:
        return False

# Checking if a variable is a finite number
def is_finite_number(x):
    if is_number(x):
        return np.isfinite(float(x))
    else:
        return False

# Checking if a variable is a NaN (not a number)
def is_nan(x):
    if is_number(x):
        return np.isnan(float(x))
    else:
        return False

# Remove any quotes, brackets (for singleton arrays), and/or invisible characters.
def remove_extra_characters(x):
    return str(x).replace('"', '').replace("'", "").replace('[', '').replace(']', '').replace(' ', '').strip()

# Sanitize boolean values.
def sanitize_boolean_value(x):
    x = remove_extra_characters(x)
    if (is_number(x) and float(x)==0) or (remove_extra_characters(str(x)) in ('False', 'false', 'FALSE', 'F', 'f')):
        return 0
    elif (is_number(x) and float(x)==1) or (remove_extra_characters(str(x)) in ('True', 'true', 'TRUE', 'T', 't')):
        return 1
    else:
        return float('nan')

def sanitize_integer_value(x):
    x = remove_extra_characters(x)
    if is_integer(x):
        return int(float(x))
    else:
        return float('nan')
    
def sanitize_scalar_value(x):
    x = remove_extra_characters(x)
    if is_number(x):
        return float(x)
    else:
        return float('nan')
def get_features(patient_metadata, recording_metadata, recording_data):
    # Extracting features from the patient metadata.
    age = get_age(patient_metadata)
    sex = get_sex(patient_metadata)
    rosc = get_rosc(patient_metadata)
    ohca = get_ohca(patient_metadata)
    vfib = get_vfib(patient_metadata)
    ttm = get_ttm(patient_metadata)

    # Using one-hot encoding 
    sex_features = np.zeros(2, dtype=int)
    if sex == 'Female':
        female = 1
        male   = 0
        other  = 0
    elif sex == 'Male':
        female = 0
        male   = 1
        other  = 0
    else:
        female = 0
        male   = 0
        other  = 1

    # Combining the patient features.
    patient_features = np.array([age, female, male, other, rosc, ohca, vfib, ttm])

    # Extracting features from the recording data and metadata.
    channels = ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'Fp1-F3',
                'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fz-Cz', 'Cz-Pz']
    num_channels = len(channels)
    num_recordings = len(recording_data)

    # Computing mean and standard deviation for each channel for each recording.
    available_signal_data = list()
    for i in range(num_recordings):
        signal_data, sampling_frequency, signal_channels = recording_data[i]
        if signal_data is not None:
            signal_data = reorder_recording_channels(signal_data, signal_channels, channels)
            available_signal_data.append(signal_data)

    if len(available_signal_data) > 0:
        available_signal_data = np.hstack(available_signal_data)
        signal_mean = np.nanmean(available_signal_data, axis=1)
        signal_std  = np.nanstd(available_signal_data, axis=1)
    else:
        signal_mean = float('nan') * np.ones(num_channels)
        signal_std  = float('nan') * np.ones(num_channels)

    # Computing the power spectral density for the delta, theta, alpha, and beta frequency bands for each channel of the most
    # recent recording.
    index = None
    for i in reversed(range(num_recordings)):
        signal_data, sampling_frequency, signal_channels = recording_data[i]
        if signal_data is not None:
            index = i
            break

    if index is not None:
        signal_data, sampling_frequency, signal_channels = recording_data[index]
        signal_data = reorder_recording_channels(signal_data, signal_channels, channels) # Reorder the channels in the signal data, as needed, for consistency across different recordings.

        delta_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency,  fmin=0.5,  fmax=8.0, verbose=False)
        theta_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency,  fmin=4.0,  fmax=8.0, verbose=False)
        alpha_psd, _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency,  fmin=8.0, fmax=12.0, verbose=False)
        beta_psd,  _ = mne.time_frequency.psd_array_welch(signal_data, sfreq=sampling_frequency, fmin=12.0, fmax=30.0, verbose=False)

        delta_psd_mean = np.nanmean(delta_psd, axis=1)
        theta_psd_mean = np.nanmean(theta_psd, axis=1)
        alpha_psd_mean = np.nanmean(alpha_psd, axis=1)
        beta_psd_mean  = np.nanmean(beta_psd,  axis=1)

        quality_score = get_quality_scores(recording_metadata)[index]
    else:
        delta_psd_mean = theta_psd_mean = alpha_psd_mean = beta_psd_mean = float('nan') * np.ones(num_channels)
        quality_score = float('nan')

    recording_features = np.hstack((signal_mean, signal_std, delta_psd_mean, theta_psd_mean, alpha_psd_mean, beta_psd_mean, quality_score))

    # Combining the features from the patient metadata and the recording data and metadata.
    features = np.hstack((patient_features, recording_features))

    return features

def train_model(data_folder, verbose):
    # Finding data files.
    if verbose >= 1:
        print('Finding the data...')

    patient_ids = find_data_folders(data_folder)
    num_patients = len(patient_ids)

    if num_patients==0:
        raise FileNotFoundError('No data was provided.')

    # Extracting the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the data...')

    features = list()
    outcomes = list()
    cpcs = list()

    for i in range(num_patients):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patients))

        # Loading data.
        patient_id = patient_ids[i]
        patient_metadata, recording_metadata, recording_data = load_data(data_folder, patient_id)

        # Extracting features.
        current_features = get_features(patient_metadata, recording_metadata, recording_data)
        features.append(current_features)

        # Extracting labels.
        current_outcome = get_outcome(patient_metadata)
        outcomes.append(current_outcome)
        current_cpc = get_cpc(patient_metadata)
        cpcs.append(current_cpc)

    features = np.vstack(features)
    outcomes = np.vstack(outcomes)
    cpcs = np.vstack(cpcs)
    
    
    # Training the models.
    if verbose >= 1:
        print('Training both the models on the data...')

    #linear regression model
    imputer = SimpleImputer().fit(features)

    # Training the models.
    features = imputer.transform(features)


    model_lr_outcome = LinearClassifierOutcome(features.shape[1])
    model_lr_cpc = LinearClassifierCPC(features.shape[1])

    criterion_outcome = nn.BCELoss()
    criterion_cpc = nn.MSELoss()

    # Creating the optimizer
    optimizer_lr_outcome = optim.SGD(model_lr_outcome.parameters(), lr=0.001)
    optimizer_lr_cpc = optim.Adam(model_lr_cpc.parameters(), lr=0.001, weight_decay=0.001)
    
    print("********************************Training of Model ********************************")
    num_epochs = 50
    # print("Feature shape: ", features.shape)
    for epoch in range(num_epochs):
        epoch_loss_outcome = 0
        epoch_loss_cpc = 0
        for i in range(len(features)):
            x = torch.tensor(features[i]).float()

            y_true_outcome = torch.tensor([outcomes.ravel()[i]]).float()
            y_true_cpc = torch.tensor([cpcs.ravel()[i]]).float()

            # Zero the gradients
            optimizer_lr_outcome.zero_grad()
            optimizer_lr_cpc.zero_grad()

            # Forwarding pass
            # print("shape of train ",x.shape)
            y_pred_outcome = model_lr_outcome(x)
            y_pred_cpc = model_lr_cpc(x)

            # Calculating loss
            # print(y_pred_outcome, y_true_outcome)
            loss_outcome = criterion_outcome(y_pred_outcome, y_true_outcome)
            loss_cpc = criterion_cpc(y_pred_cpc, y_true_cpc)

            # #Rounding off predictions here rather than the forward method
            # y_pred_outcome = torch.round(y_pred_outcome)
            loss_outcome.backward()
            loss_cpc.backward()

            optimizer_lr_outcome.step()
            optimizer_lr_cpc.step()

            epoch_loss_outcome += loss_outcome.item()
            epoch_loss_cpc += loss_cpc.item()

        print(f"Epoch {epoch}: Loss outcome = {epoch_loss_outcome/len(features)} Loss = {epoch_loss_cpc/len(features)}")

    if verbose >= 1:
        print('Done.')

    return imputer, model_lr_outcome, model_lr_cpc



def run_models(imputer, outcome_model, cpc_model, data_folder, patient_id, verbose):
    # Loading data.
    patient_metadata, recording_metadata, recording_data = load_data(data_folder, patient_id)

    # Extracting features.
    features = get_features(patient_metadata, recording_metadata, recording_data)
    features = features.reshape(1, -1)

    # Imputing missing data.
    features = imputer.transform(features)
    features = torch.tensor(features, dtype=torch.float32)

    # Applying models to features.
    outcome_logits = outcome_model(features)
    outcome_probability = torch.sigmoid(outcome_logits)  # Applying sigmoid to the output logits to get probabilities
    outcome = (outcome_probability > 0.5).float()  # Getting the predicted class

    cpc = cpc_model(features)

    cpc = cpc.detach().numpy()
    cpc = np.clip(cpc, 1, 5)

    poor_outcome_probability = 1 - outcome_probability.item()  # getting the probability of class 0 ("Poor")

    return outcome.item(), poor_outcome_probability, cpc.item()

def run_model(imputer, outcome_model, cpc_model, data_folder, output_folder, allow_failures, verbose):
    # Loading the model.
    if verbose >= 1:
        print('Loading the models...')


    # Finding the data.
    if verbose >= 1:
        print('Finding the data...')

    patient_ids = find_data_folders(data_folder)
    num_patients = len(patient_ids)

    if num_patients==0:
        raise Exception('No data was provided.')

    if verbose >= 1:
        print('Running the models on the data...')

    # Iterating over the patients.
    outcome_binary_list = []
    outcome_probability_list = []
    cpc_list = []

    for i in range(num_patients):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patients))

        patient_id = patient_ids[i]

        try:
            outcome_binary, outcome_probability, cpc = run_models(imputer, outcome_model, cpc_model, data_folder, patient_id, verbose) ### Teams: Implement this function!!!
            
            outcome_binary_list.append(outcome_binary)
            # outcome_probability_list.append(outcome_probability)
            cpc_list.append(cpc)

        except:
            if allow_failures:
                if verbose >= 2:
                    print('... failed.')
                outcome_binary, outcome_probability, cpc = float('nan'), float('nan'), float('nan')
                outcome_binary_list.append(0.0)
                # outcome_probability_list.append(0.0)
                cpc_list.append(0.0)
            else:
                raise

    

    if verbose >= 1:
        print('Done.')
    return outcome_binary_list, cpc_list

def getCPC(path):
    folders = find_data_folders(path)
    pat_cpc = dict()
    for i in range(len(folders)):
        patient = folders[i]
        patient_metadata = load_clinic_data(path, patient)
        pat_cpc[patient] = get_cpc(patient_metadata)

    return pat_cpc

def getOutcome(path):
    folders = find_data_folders(path)
    pat_cpc = dict()
    for i in range(len(folders)):
        patient = folders[i]
        patient_metadata = load_clinic_data(path, patient)
        pat_cpc[patient] = get_outcome(patient_metadata)
    return pat_cpc

if __name__ == '__main__':
    if not (len(sys.argv) == 3 or len(sys.argv) == 4):
        raise Exception('Include the data and model folders as arguments, e.g., python train_model.py data model.')

    # Define the data and model foldes.
    data_folder = sys.argv[1]
    output_folder = sys.argv[2]

    #python train_model.py datafolder 
    if len(sys.argv)==4 and is_integer(sys.argv[3]):
        verbose = int(sys.argv[3])
    else:
        verbose = 1

    imputer, model_lr_outcome, model_lr_cpc= train_model(data_folder, verbose) ### Teams: Implement this function!!!

    allow_failures = False


    if len(sys.argv)==5 and is_integer(sys.argv[4]):
        verbose = int(sys.argv[4])
    else:
        verbose = 1

    predicted_outcome, predicted_cpc = run_model(imputer, model_lr_outcome, model_lr_cpc, data_folder, output_folder, allow_failures, verbose)
    
    original_cpc = getCPC(data_folder)
    original_outcome = getOutcome(data_folder)

    x_outcome = [value for value in original_outcome.values() if isinstance(value, (int, float))]
    # y_outcome = [value for value in predicted_outcome if isinstance(value, (int, float))]

    x_cpc = [value for value in original_cpc.values() if isinstance(value, (int, float))]
    # y_cpc = [value for value in predicted_cpc if isinstance(value, (int, float))]
    # print("____________________________________________________")

    # print(x_outcome, predicted_outcome)
    # print("____________________________________________________")
    accuracy = accuracy_score(x_outcome, predicted_outcome)
    f_measure = f1_score(x_outcome, predicted_outcome)

    # print(x_cpc, predicted_cpc)
    mse = mean_squared_error(x_cpc, predicted_cpc)
    mae = mean_absolute_error(x_cpc, predicted_cpc)

    output_string = \
        'Outcome Accuracy: {:.3f}\n'.format(accuracy) + \
        'Outcome F-measure: {:.3f}\n'.format(f_measure) + \
        'CPC MSE: {:.3f}\n'.format(mse) + \
        'CPC MAE: {:.3f}\n'.format(mse)
    
    print(output_string)
