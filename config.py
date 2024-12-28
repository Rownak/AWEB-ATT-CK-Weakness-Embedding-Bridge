import os
ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
DATASETS = ['ics_attack']
ATTACK_DATASET = 'ics_attack'
CWE_DATASET = 'cwec_v4.12'
# # Do not change
LLM_FT_EPOCH = 10
ATTACK_DIR = ROOT_PATH+"/datasets/{}/".format(ATTACK_DATASET)
CWE_DIR = ROOT_PATH+"/datasets/{}/".format(CWE_DATASET)
OUTPUT_DIR = ROOT_PATH+"/output/{}/".format(ATTACK_DATASET)
EMBEDDING_DIR = OUTPUT_DIR+"embeddings/"
DESCRIPTION_FILE = ATTACK_DIR+"doc_id_to_desc_only.json"
