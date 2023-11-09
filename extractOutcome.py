

import json
from team_code import *
from helper_code import *

def getOutcome(path):
    folders = find_data_folders(path)
    pat_cpc = dict()
    for i in range(len(folders)):
        patient = folders[i]
        patient_metadata = load_clinic_data(path, patient)
        pat_cpc[patient] = get_outcome(patient_metadata)
    return pat_cpc
if __name__ == '__main__':
    parent_folder = sys.argv[1]
    output_folder = sys.argv[2]

    patient_cpc = getOutcome(parent_folder)

    json_str = json.dumps(patient_cpc)

    with open(output_folder, "w") as file:
        file.write(json_str)

# python extractCPCs.py i-care\training cpc_original
