from Base import *
from Condition import *
from Config import *
import json
import numpy as np
import random
import simpy

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
                      env=env,
                      object_type="UE")

        self.coverage_info = coverage_info
        self.DURATION = coverage_info.shape[2]
        self.serving_satellite = None
        self.satellites = None
        self.oracle = oracle

        self.messageQ = simpy.Store(env)
        self.state = ACTIVE
        # This is an object 'UE_condition'
        self.condition = None

        # Logs
        self.serving_satellite_history = []

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
                covered_satellites = np.where(self.coverage_info[self.identity, :, now] == 1)[0]
                possible_candidates = np.delete(covered_satellites,
                                                np.where(covered_satellites == self.serving_satellite.identity))
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
                self.serving_satellite = self.satellites[data['from']]
                self.serving_satellite_history.append(data['from'])
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

    def decide_best_action(self, available_conditions, left_conditions):
        time = self.env.now
        sat_id = -1
        if len(left_conditions) == 0:
            condition = random.choice(available_conditions)
            sat_id = condition.satid
        else:
            prob = 0.5
            if random.random() < prob:
                condition = random.choice(available_conditions)
                sat_id = condition.satid
        return sat_id

    def select_candidates(self, candidates):
        if len(candidates) > NUMBER_CANDIDATE:
            selected_candidates = random.sample(candidates, NUMBER_CANDIDATE)
            return selected_candidates
        else:
            return candidates
