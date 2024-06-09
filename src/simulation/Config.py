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
PATH_SWITCH_REQUEST = "PATH_SWITCH_REQUEST"
PATH_SWITCH_REQUEST_ACK = "PATH_SWITCH_REQUEST_ACK"
UE_CONTEXT_RELEASE = "UE_CONTEXT_RELEASE"

NUMBER_CANDIDATE = 3
WINDOW_SIZE = 200
max_access_slots = WINDOW_SIZE
oracle_assignment = False
oracle_simulation = False

SOURCE_DECISION_LONGEST = "SD_LONGEST"
SOURCE_DECISION_EARLIEST = "SD_EARLIEST"
SOURCE_DECISION_RANDOM = "SD_RANDOM"

CANDIDATE_EARLIEST = "C_EARLIEST"
CANDIDATE_RANDOM = "C_RANDOM"

SOURCE_SELECTION_LONGEST = "SS_LONGEST"
SOURCE_SELECTION_RANDOM = "SS_RANDOM"

# sys.argv = ['notebook', SOURCE_DECISION_LONGEST, CANDIDATE_RANDOM, SOURCE_SELECTION_LONGEST, '1'] # NOTE THIS WORK IN NOTEBOOK
# Modify this list if you have anything to compare
SOURCE_DECISION_ALG = sys.argv[1]
CANDIDATE_ALG = sys.argv[2]
SOURCE_SELECTION_ALG = sys.argv[3]
max_access_opportunity = int(sys.argv[4])
RESULT_PATH = "./result/" + SOURCE_DECISION_ALG + '|' + CANDIDATE_ALG + '|' + SOURCE_SELECTION_ALG + '|' + str(max_access_opportunity) + '.pkl'
