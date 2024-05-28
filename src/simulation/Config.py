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
WINDOW_SIZE = 100
max_access_opportunity = 4
max_access_slots = WINDOW_SIZE
oracle_assignment = False
oracle_simulation = False

SOURCE_ALG_LONGEST = "S_LONGEST"
SOURCE_ALG_EARLIEST = "S_EARLIEST"
SOURCE_ALG_RANDOM = "S_RANDOM"
SOURCE_ALG_OUR = "CANDIDATE_ALG_OUR"

CANDIDATE_ALG_EARLIEST = "C_EARLIEST"
CANDIDATE_ALG_RANDOM = "C_RANDOM"
CANDIDATE_ALG_OUR = "CANDIDATE_ALG_OUR"

SOURCE_ALG = sys.argv[1]
CANDIDATE_ALG = sys.argv[2]
RESULT_PATH = "./result/" + SOURCE_ALG + '|' + CANDIDATE_ALG + '.pkl'
