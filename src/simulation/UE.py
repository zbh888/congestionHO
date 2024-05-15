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
                covered_satellites_future = np.where(self.coverage_info[self.identity, :, min(now+25, self.DURATION-1)] == 1)[0]
                possible_candidates = np.intersect1d(covered_satellites_now, covered_satellites_future)
                candidates = self.select_candidates(possible_candidates)
                assert (len(candidates) > 0)
                source = self.satellites[self.serving_satellite.identity]
                data = {
                    "task": MEASUREMENT_REPORT,
                    "candidates": candidates.tolist(),
                }
                self.send_message(
                    msg=data,
                    to=source
                )
            # ================================================
            if self.state == RRC_CONFIGURED:
                satid = self.determine_if_access()
                if satid != -1:
                    target = self.satellites[satid]
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
                conditions = data['conditions']
                self.condition = UE_condition(conditions, self.identity, self.env.now)
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
                self.applied_delay_history.append(self.condition.conditions[target_id].access_delay)
                self.condition = None

    def determine_if_access(self):
        assert (self.condition is not None)
        assert (self.state == RRC_CONFIGURED)
        available_conditions, left_conditions = self.condition.available_access_conditions(self.env.now)
        if len(available_conditions) == 0:
            return -1
        else:
            satid = self.decide_best_action(available_conditions, left_conditions)
            if satid == -1:
                return -1
            else:
                return satid

    # This will use orcale if applicable
    def decide_best_action(self, available_conditions, left_conditions):
        if self.oracle is not None:
            targetid = self.oracle.query_next_satellite(self.identity, self.serving_satellite.identity)

        if self.oracle is not None and targetid != -1:
            found = False
            for condition in available_conditions:
                if targetid == condition.satid:
                    found = True
            if not found and len(left_conditions) == 0:
                raise AssertionError("The UE did not receive condition from oracle arranged target satellite")
            if found:
                return targetid
            else:
                return -1
        else:
            sat_id = -1
            if len(left_conditions) == 0:
                condition = random.choice(available_conditions)
                # This choose the target
                sat_id = condition.satid
            else:
                prob = 0.5
                if random.random() < prob:
                    condition = random.choice(available_conditions)
                    # This choose the target
                    sat_id = condition.satid
            return sat_id

    # This will use orcale if applicable
    def select_candidates(self, candidates):
        if self.oracle is not None:
            targetid = self.oracle.query_next_satellite(self.identity, self.serving_satellite.identity)

        if self.oracle is not None and targetid != -1:
            if targetid not in candidates:
                raise AssertionError(
                    f"UE {self.identity} at time: {self.env.now}, serving_satellite {self.serving_satellite.identity}, target {targetid} not in the candidate set")
            if len(candidates) > NUMBER_CANDIDATE:
                candidates2 = copy.deepcopy(candidates)
                candidates2.remove(targetid)
                selected_candidates = random.sample(candidates2, NUMBER_CANDIDATE - 1)
                selected_candidates.append(targetid)
                return selected_candidates
            else:
                return candidates


        else:
            if len(candidates) > NUMBER_CANDIDATE:
                selected_candidates = random.sample(candidates, NUMBER_CANDIDATE)
                return selected_candidates
            else:
                return candidates
