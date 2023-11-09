
import os, numpy as np, scipy as sp, scipy.io
# Find the folders with data files.
def find_data_folders(root_folder):
    data_folders = list()
    for x in os.listdir(root_folder):
        data_folder = os.path.join(root_folder, x)
        if os.path.isdir(data_folder):
            data_folders.append(x)
    return sorted(data_folders)
# Load text file as a string.
def load_text_file(filename):
    with open(filename, 'r') as f:
        data = f.read()
    return data
# Get the recording IDs column from the patient data.
def get_recording_ids(string):
    return get_column(string, 'Record', str)

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
def remove_extra_characters(x):
    return str(x).replace('"', '').replace("'", "").replace('[', '').replace(']', '').replace(' ', '').strip()
# Sanitize integer values.
def sanitize_integer_value(x):
    x = remove_extra_characters(x)
    if is_integer(x):
        return int(float(x))
    else:
        return float('nan')

# Santize scalar values.
def sanitize_scalar_value(x):
    x = remove_extra_characters(x)
    if is_number(x):
        return float(x)
    else:
        return float('nan')
# Sanitize boolean values.
def sanitize_boolean_value(x):
    x = remove_extra_characters(x)
    if (is_number(x) and float(x)==0) or (remove_extra_characters(str(x)) in ('False', 'false', 'FALSE', 'F', 'f')):
        return 0
    elif (is_number(x) and float(x)==1) or (remove_extra_characters(str(x)) in ('True', 'true', 'TRUE', 'T', 't')):
        return 1
    else:
        return float('nan')
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

# Check if a variable is a boolean or represents a boolean.
def is_boolean(x):
    if (is_number(x) and float(x)==0) or (remove_extra_characters(x) in ('False', 'false', 'FALSE', 'F', 'f')):
        return True
    elif (is_number(x) and float(x)==1) or (remove_extra_characters(x) in ('True', 'true', 'TRUE', 'T', 't')):
        return True
    else:
        return False

# Check if a variable is a finite number or represents a finite number.
def is_finite_number(x):
    if is_number(x):
        return np.isfinite(float(x))
    else:
        return False

# Check if a variable is a NaN (not a number) or represents a NaN.
def is_nan(x):
    if is_number(x):
        return np.isnan(float(x))
    else:
        return False
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

def load_challenge_data(data_folder, patient_id):
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
