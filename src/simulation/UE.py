from Base import *
from Condition import *
from Config import *
import json
import numpy as np
import random
import copy


class UE(Base):
    def __init__(self,
                 identity,
                 position_x,
                 position_y,
                 coverage_info,
                 oracle,
                 env):

        # Config Initialization
        Base.__init__(self,
                      identity=identity,
                      position_x=position_x,
                      position_y=position_y,
                      coverage_info=coverage_info,
                      env=env,
                      object_type="UE")

        self.serving_satellite = None
        self.oracle = oracle

        self.state = ACTIVE
        # This is an object 'UE_condition'
        self.condition = None

        # Logs
        self.serving_satellite_history = []
        self.applied_delay_history = []

        # Running Process
        env.process(self.init())
        self.env.process(self.action_monitor())
        self.env.process(self.handle_messages())

    # ====== UE functions ======
    def action_monitor(self):
        while True:
            now = self.env.now
            # ================================================
            next = now + 1
            # needs to send measurement report
            if (next < self.DURATION and self.state == ACTIVE
                    and self.coverage_info[self.identity, self.serving_satellite.identity, now] == 1
                    and self.coverage_info[self.identity, self.serving_satellite.identity, next] == 0):
                covered_satellites_now = np.where(self.coverage_info[self.identity, :, now] == 1)[0]
                covered_satellites_future = \
                    np.where(self.coverage_info[self.identity, :, min(now + WINDOW_SIZE, self.DURATION - 1)] == 1)[0]
                candidates = np.intersect1d(covered_satellites_now, covered_satellites_future)
                assert (len(candidates) >= NUMBER_CANDIDATE)
                source = self.satellites[self.serving_satellite.identity]
                candidates_utilities = []
                for satid in candidates:
                    candidates_utilities.append(self.estimate_serving_length(satid))
                data = {
                    "task": MEASUREMENT_REPORT,
                    "candidates": candidates.tolist(),
                    "utility": candidates_utilities,
                }
                self.send_message(
                    msg=data,
                    to=source
                )
            # ================================================
            if self.state == RRC_CONFIGURED:
                if self.determine_if_access():
                    target = self.satellites[self.condition.targetid]
                    data = {
                        "task": RANDOM_ACCESS,
                    }
                    self.send_message(
                        msg=data,
                        to=target,
                    )
            # ================================================
            yield self.env.timeout(1)

    def handle_messages(self):
        while True:
            msg = yield self.messageQ.get()
            data = json.loads(msg)
            assert (data['to'] == self.identity)
            task = data['task']
            # ================================================
            if task == RRC_RECONFIGURATION:
                source = self.satellites[data['from']]
                # conditions = data['conditions'] # I expect we don't need it
                targetid = data['suggested_target']
                corresponding_delay = data['corresponding_delay']
                self.condition = UE_condition(targetid, corresponding_delay, self.env.now)
                self.state = RRC_CONFIGURED
                data = {
                    "task": RRC_RECONFIGURATION_COMPLETE,
                }
                self.send_message(
                    msg=data,
                    to=source,
                )
                # ================================================
            if task == RANDOM_ACCESS_RESPONSE:
                # Cleanup for the UE
                self.state = ACTIVE
                target_id = data['from']
                self.serving_satellite = self.satellites[target_id]
                self.serving_satellite_history.append(target_id)
                self.applied_delay_history.append(self.condition.delay)
                self.condition = None

    def determine_if_access(self):
        return self.condition.access_time == self.env.now

    def prepare_candidates_utilities(self, candidates):
        candidate_utility = []
        for satid in candidates:
            serving_time = self.estimate_serving_length(satid)
            if serving_time > WINDOW_SIZE:  # TODO This may be adjusted to min_serving_time
                candidate_utility.append((satid, self.estimate_serving_length(satid)))
        return candidates.tolist(), candidate_utility

    # This will use orcale if applicable
    # def select_candidates(self, candidates):
    #     if self.oracle is not None:
    #         targetid = self.oracle.query_next_satellite(self.identity, self.serving_satellite.identity)
    #
    #     if self.oracle is not None and targetid != -1:
    #         if targetid not in candidates:
    #             raise AssertionError(
    #                 f"UE {self.identity} at time: {self.env.now}, serving_satellite {self.serving_satellite.identity}, target {targetid} not in the candidate set")
    #         if len(candidates) > NUMBER_CANDIDATE:
    #             candidates2 = copy.deepcopy(candidates)
    #             candidates2.remove(targetid)
    #             selected_candidates = random.sample(candidates2, NUMBER_CANDIDATE - 1)
    #             selected_candidates.append(targetid)
    #             return np.array(selected_candidates)
    #         else:
    #             return candidates
    #
    #
    #     else:
    #         if len(candidates) > NUMBER_CANDIDATE:
    #             candidate_utility = []
    #             for satid in candidates:
    #                 serving_time = self.estimate_serving_length(satid)
    #                 if serving_time > WINDOW_SIZE:  # TODO This may be adjusted to min_serving_time
    #                     candidate_utility.append((satid, self.estimate_serving_length(satid)))
    #             sorted_list = sorted(candidate_utility, key=lambda x: -x[1])
    #             # find best candidates
    #             if UE_ALG == UE_ALG_LONGEST:
    #                 selected_candidates = [x[0] for x in sorted_list][:NUMBER_CANDIDATE]
    #             # random
    #             elif UE_ALG == UE_ALG_RANDOM:
    #                 selected_candidates = random.sample(candidates.tolist(), NUMBER_CANDIDATE)
    #             else:
    #                 print(UE_ALG)
    #             return np.array(selected_candidates)
    #         else:
    #             return candidates

    def estimate_serving_length(self, satid):
        # The returned value means from the current time + {return}, it will perform handover.
        x, = np.where(self.coverage_info[self.identity, satid, self.env.now:] == 0)
        if len(x) == 0:
            return random.randint(4000, 6000)
        else:
            return int(x[0] - 1)
