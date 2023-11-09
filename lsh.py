# Hiteshwar Singh(115076518), Praveen Raj(114999920), Shwetali Rane(115320783), Ritvik Patil(115131064) 

from pyspark import SparkConf, SparkContext
import numpy as np
import random
import mmh3
from data_retrieve import *
import sys

def start_spark_context():
    conf = (SparkConf()
             .setMaster("local")
             .setAppName("PatientSimilarity"))
    sc = SparkContext(conf = conf)
    return sc

sc = start_spark_context()

def getCombinedRDD(path):
  folders = find_data_folders(path)
  all_records = []
  for i in range(len(folders)):
      patient = folders[i]
      patient_metadata, recording_metadata, recording_data = load_challenge_data(path, patient)
      records = []

      for i in recording_data[4:]:
          if i[0] is not None:
            records.append(i[0].T)
      if len(records)!=0:
        records_st = np.vstack(records)
        all_records.append(records_st.tolist())
      else:
        all_records.append([])
      print(patient," stored in all_records")
  combined_rdd = sc.parallelize(zip(folders, all_records))

  return combined_rdd

def getHashList(shingles, hashes):
    hashed = []
    for h in hashes:
        val = np.inf
        for shingle in shingles:
            hash_val = mmh3.hash(str(shingle), seed=h, signed=False)
            val = min(val, hash_val)
        hashed.append(val)
    return hashed

def getRandPatients(path, count):
    folders = find_data_folders(path)
    if count > len(folders):
        raise ValueError("Count cannot be greater than the array length.")
    random_values = random.sample(folders, count)
    return random_values

def jaccard_similarity(set1, set2):
    intersection = len(set(set1).intersection(set2))
    union = (len(set1) + len(set2)) - intersection
    return float(intersection) / union

def find_similar_patients(path, num_hashes, num_target_patients):
    combined_rdd = getCombinedRDD(path)

    # Compute hash signatures for each patient
    signatures = combined_rdd.flatMap(lambda x: [(x[0], getHashList(x[1], range(num_hashes)))]) 

    target_patients = getRandPatients(path, num_target_patients)
    target_patient_signatures = signatures.filter(lambda x: x[0] in target_patients)
    check_patient_signatures = signatures.filter(lambda x: x[0] not in target_patients)

    numBands = 50
    perBand = num_hashes // numBands

    # Apply banding and hash the bands
    target_patient_bands = target_patient_signatures.flatMap(lambda x: [((x[0], i//perBand), (i, x[1][i])) for i in range(num_hashes)])\
                                                     .reduceByKey(lambda x,y: str(x)+str(y))\
                                                     .map(lambda x: (((x[0][1], mmh3.hash(x[1], seed=x[0][1], signed=False)), x[0][0])))

    check_patient_bands = check_patient_signatures.flatMap(lambda x: [((x[0], i//perBand), (i, x[1][i])) for i in range(num_hashes)])\
                                                   .reduceByKey(lambda x,y: str(x)+str(y))\
                                                   .map(lambda x: (((x[0][1], mmh3.hash(x[1], seed=x[0][1], signed=False)), x[0][0])))

    # Find candidate pairs by joining on the band hash
    candidate_pairs = target_patient_bands.join(check_patient_bands)\
                                          .map(lambda x: x[1])\
                                          .reduceByKey(lambda x,y : x)

    # Collect the signatures for the candidate pairs to avoid recomputation
    signature_dict = signatures.collectAsMap()

    # Calculate Jaccard similarity for candidate pairs and filter pairs with similarity > 0.8
    similar_pairs = candidate_pairs.filter(lambda x: jaccard_similarity(set(signature_dict[x[0]]), set(signature_dict[x[1]])) > 0.8).collect()

    return similar_pairs

print(similarities)
if __name__ == "__main__":
    path = sys.argv[1]
    
    num_hashes = 1000
    num_target_patients = 5

    similarities = find_similar_patients(path, num_hashes, num_target_patients)

    print(similarities)
    

