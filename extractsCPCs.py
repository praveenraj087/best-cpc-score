# Hiteshwar Singh(115076518), Praveen Raj(114999920), Shwetali Rane(115320783), Ritvik Patil(115131064) 

import json
from team_code import *
from helper_code import *

def getCPC(path):
    folders = find_data_folders(path)
    pat_cpc = dict()
    for i in range(len(folders)):
        patient = folders[i]
        patient_metadata = load_clinic_data(path, patient)
        pat_cpc[patient] = extractCPC(patient_metadata)
    return pat_cpc
if __name__ == '__main__':
    parent_folder = sys.argv[1]
    output_folder = sys.argv[2]

    patient_cpc = getCPC(parent_folder)

    json_str = json.dumps(patient_cpc)

    with open(output_folder, "w") as file:
        file.write(json_str)

# python extractCPCs.py i-care\training cpc_original
