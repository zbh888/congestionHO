import sys

ACTIVE = "ACTIVE"
RRC_CONFIGURED = "RRC_CONFIGURED"

MEASUREMENT_REPORT = "MEASUREMENT_REPORT"
HANDOVER_REQUEST = "HANDOVER_REQUEST"
HANDOVER_RESPONSE = "HANDOVER_RESPONSE"
RRC_RECONFIGURATION = "RRC_RECONFIGURATION"
RANDOM_ACCESS = "RANDOM_ACCESS"
RANDOM_ACCESS_RESPONSE = "RANDOM_ACCESS_RESPONSE"
RRC_RECONFIGURATION_COMPLETE = "RRC_RECONFIGURATION_COMPLETE"
HANDOVER_SUCCESS = "HANDOVER_SUCCESS"
SN_STATUS_TRANSFER = "SN_STATUS_TRANSFER"
HANDOVER_CANCEL = "HANDOVER_CANCEL"

NUMBER_CANDIDATE = 3
WINDOW_SIZE = 200
max_access_slots = WINDOW_SIZE
oracle_assignment = False
oracle_simulation = False

SOURCE_ALG_OUR = "SOURCE_ALG_OUR"

CANDIDATE_ALG_OUR = "CANDIDATE_ALG_OUR"

UE_ALG_LONGEST = "UE_LONGEST"
UE_ALG_RANDOM = "UE_RANDOM"

# Modify this list if you have anything to compare
sys.argv = ['notebook', SOURCE_ALG_OUR, CANDIDATE_ALG_OUR, UE_ALG_LONGEST, '1'] # NOTE THIS WORK IN NOTEBOOK
SOURCE_ALG = sys.argv[1]
CANDIDATE_ALG = sys.argv[2]
UE_ALG = sys.argv[3]
max_access_opportunity = int(sys.argv[4])
RESULT_PATH = "./result/" + SOURCE_ALG + '|' + CANDIDATE_ALG + '|' + UE_ALG + '|' + str(max_access_opportunity) + '.pkl'

UE_HANDOVER_SIGNALLING_COUNT_ON_SOURCE = 5
TARGET_HANDOVER_SUCCESS_SIGNALLING_COUNT_ON_SOURCE = 1
SOURCE_HANDOVER_REQUEST_SIGNALLING_COUNT_ON_CANDIDATE = 2 # Assuming candidate will be target.
