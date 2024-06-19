import random

from Base import *
from Condition import *
from Config import *
from Counter import *
from Queue import *
import numpy as np


class Satellite(Base):
    def __init__(self,
                 identity,
                 position_x,
                 position_y,
                 height,
                 coverage_r,
                 velocity,
                 sind,
                 cosd,
                 coverage_info,
                 max_access_opportunity,
                 max_access_slots,
                 oracle,
                 env):

        # Config Initialization
        Base.__init__(self,
                      identity=identity,
                      position_x=position_x,
                      position_y=position_y,
                      coverage_info=coverage_info,
                      env=env,
                      object_type="Satellite")

        self.coverage_r = coverage_r
        self.AMF = None
        self.height = height
        self.velocity = velocity
        self.sind = sind
        self.cosd = cosd
        self.UEs = None
        self.access_Q = Queue(max_access_opportunity, max_access_slots)
        self.current_assigned_slot = None
        self.oracle = oracle
        self.record_max_delay = 0  # may be removed, recording the largest delay returned, not apply to RANDOM
        self.reservation_count = 0

        # === source function ===
        # condition_record[ueid] stores the received conditions from candidates ([Sat_condition, ..., Sat_condition])
        self.condition_record = {}
        # candidates_record[ueid] stores candidates
        self.candidates_record = {}

        self.load_aware = {}  # satid -> (time, (priority, load, potential_load))
        self.predicted_my_load = [0] * (self.DURATION + 5000)  # They know the future
        self.predicted_my_load_potential = [0] * (self.DURATION + 5000)  # So, no one knows the future
        self.within_one_slot_load_priority = 0

        # === target function ===
        # takeover_condition_record[ueid] stores 
        self.takeover_condition_record = {}

        self.counter = counter(self.DURATION)

        # Running Process
        self.env.process(self.init())
        self.env.process(self.action_monitor())
        self.env.process(self.handle_messages())

    # ====== Satellite functions ======
    def prepare_my_load_prediction(self):

        return (self.within_one_slot_load_priority,
                self.predicted_my_load[self.env.now:self.env.now + self.access_Q.max_access_slots + 1],
                self.predicted_my_load_potential[self.env.now:self.env.now + self.access_Q.max_access_slots + 1])

    def prepare_other_load_prediction(self, satid):
        if satid not in self.load_aware:
            return (0, [], [])
        else:
            candidate_time = self.load_aware[satid][0]
            candidate_priority = self.load_aware[satid][1][0]
            candidate_load = self.load_aware[satid][1][1][self.env.now - candidate_time:]
            candidate_load_potential = self.load_aware[satid][1][2][self.env.now - candidate_time:]
            return (candidate_priority, candidate_load, candidate_load_potential)

    def increment_my_load(self, time, amount):
        # print(f"{self.identity},{self.env.now} [{time}, + {amount}]: real load")
        self.within_one_slot_load_priority += 1
        self.predicted_my_load[time] += amount

    def increment_my_load_potential(self, time, amount):
        # print(f"{self.identity},{self.env.now} [{time}, + {amount}]: potential load")
        self.within_one_slot_load_priority += 1
        self.predicted_my_load_potential[time] += amount

    def decrease_my_load(self, time, amount):
        # print(f"{self.identity},{self.env.now} [{time}, - {amount}]: real load")
        self.within_one_slot_load_priority += 1
        self.predicted_my_load[time] -= amount
        assert (self.predicted_my_load[time] >= 0)

    def decrease_my_load_potential(self, time, amount):
        # print(f"{self.identity},{self.env.now} [{time}, - {amount}]: potential load")
        self.within_one_slot_load_priority += 1
        self.predicted_my_load_potential[time] -= amount
        assert (self.predicted_my_load_potential[time] >= 0)

    def update_other_priority_load(self, satid, priority_load):
        if satid not in self.load_aware:
            self.load_aware[satid] = (self.env.now, priority_load)
        else:
            old_priority_load = self.prepare_other_load_prediction(satid)
            old_priority = old_priority_load[0]
            old_load_length = len(old_priority_load[1])
            new_priority = priority_load[0]
            new_load_length = len(priority_load[1])
            if new_load_length == old_load_length:
                if new_priority > old_priority:
                    self.load_aware[satid] = (self.env.now, priority_load)
            elif new_load_length > old_load_length:
                self.load_aware[satid] = (self.env.now, priority_load)

    def action_monitor(self):
        while True:
            yield self.env.timeout(0.999999)
            self.current_assigned_slot, reservation_count = self.access_Q.shift()
            self.reservation_count += reservation_count
            self.within_one_slot_load_priority = 0

    def handle_messages(self):
        while True:
            msg = yield self.messageQ.get()
            data = json.loads(msg)
            assert (data['to'] == self.identity)
            now = self.env.now
            task = data['task']
            if task not in [MEASUREMENT_REPORT, RANDOM_ACCESS, RRC_RECONFIGURATION_COMPLETE, PATH_SWITCH_REQUEST_ACK]:
                self.load_aware[data['from']] = (self.env.now, data['priority_load'])
            self.counter.increment(task, now)
            # ================================================ Source
            if task == MEASUREMENT_REPORT:
                ueid = data['from']
                candidates = data['candidates']
                assert (ueid not in self.condition_record)
                assert (ueid not in self.candidates_record)
                candidates, utilities = self.candidates_selection(candidates, data['utility'])
                self.condition_record[ueid] = []
                self.candidates_record[ueid] = candidates
                for satid in candidates:
                    target_satellite = self.satellites[satid]
                    data = {
                        "task": HANDOVER_REQUEST,
                        "ueid": ueid,
                        "candidates": candidates,
                        "utility": utilities,
                        "priority_load": self.prepare_my_load_prediction(),
                        "candidates_priority_load": [self.prepare_other_load_prediction(c_satid) for c_satid in
                                                     candidates]
                    }
                    self.send_message(
                        msg=data,
                        to=target_satellite
                    )
            # ================================================ Target + Candidate
            elif task == HANDOVER_REQUEST:
                source_id = data['from']
                requested_satellite = self.satellites[source_id]
                ueid = data['ueid']
                candidates = data['candidates']
                utilities = data['utility']
                candidates_priority_loads = data['candidates_priority_load']
                for candidate_id, candidate_priority_load in zip(candidates, candidates_priority_loads):
                    self.update_other_priority_load(candidate_id, candidate_priority_load)
                condition = self.prepare_condition(ueid, source_id, candidates, utilities)
                self.increment_my_load_potential(self.env.now + condition.access_delay,
                                                 SOURCE_HANDOVER_REQUEST_SIGNALLING_COUNT_ON_CANDIDATE)
                self.increment_my_load_potential(self.env.now + condition.ue_utility,
                                                 UE_HANDOVER_SIGNALLING_COUNT_ON_SOURCE)
                data = {
                    "task": HANDOVER_RESPONSE,
                    "ueid": ueid,
                    "condition": condition.toJSON(),
                    'priority_load': self.prepare_my_load_prediction(),
                }
                self.send_message(
                    msg=data,
                    to=requested_satellite
                )
                self.takeover_condition_record[ueid] = condition
            # ================================================ Source
            elif task == HANDOVER_RESPONSE:
                ueid = data["ueid"]
                condition = data["condition"]
                self.condition_record[ueid].append(condition)
                if len(self.condition_record[ueid]) == len(self.candidates_record[ueid]):
                    best_targetid, delay = self.decide_best_target(ueid)
                    data = {
                        "task": RRC_RECONFIGURATION,
                        "conditions": self.condition_record[ueid],
                        "suggested_target": best_targetid,
                        "corresponding_delay": delay,
                    }
                    self.increment_my_load(self.env.now + delay, TARGET_HANDOVER_SUCCESS_SIGNALLING_COUNT_ON_SOURCE)
                    UE_who_requested = self.UEs[ueid]
                    self.send_message(
                        msg=data,
                        to=UE_who_requested
                    )
            # ================================================ Target
            elif task == RANDOM_ACCESS:
                # Response to UE
                ueid = data['from']
                expected_access_time, expected_leaving_time = self.estimated_access_handover_precise_time(ueid)
                self.increment_my_load(expected_leaving_time, UE_HANDOVER_SIGNALLING_COUNT_ON_SOURCE)
                self.decrease_my_load_potential(expected_leaving_time, UE_HANDOVER_SIGNALLING_COUNT_ON_SOURCE)
                assert (self.current_assigned_slot.include(ueid))
                assert (self.current_assigned_slot.time == self.env.now)
                takeover_condition = self.takeover_condition_record[ueid]
                UE_who_requested = self.UEs[ueid]
                data = {
                    "task": RANDOM_ACCESS_RESPONSE,
                }
                self.send_message(
                    msg=data,
                    to=UE_who_requested
                )

                # Response to source Satellite
                sourceid = takeover_condition.sourceid
                source = self.satellites[sourceid]
                data = {
                    "task": HANDOVER_SUCCESS,
                    "ueid": ueid,
                    'priority_load': self.prepare_my_load_prediction()
                }
                self.send_message(
                    msg=data,
                    to=source
                )
                # upon receiving random access, the target delete the condition record and take over UE
                del self.takeover_condition_record[ueid]
            # ================================================ Source
            elif task == HANDOVER_SUCCESS:
                # send SN status transfer
                target_id = data['from']
                target = self.satellites[target_id]
                ueid = data['ueid']
                data = {
                    "task": SN_STATUS_TRANSFER,
                    "ueid": ueid,
                    'priority_load': self.prepare_my_load_prediction()
                }
                self.send_message(
                    msg=data,
                    to=target
                )

                # cancel other candidates
                for candidateid in self.candidates_record[ueid]:
                    if candidateid != target_id:
                        candidate = self.satellites[candidateid]
                        data = {
                            "task": HANDOVER_CANCEL,
                            "ueid": ueid,
                            'priority_load': self.prepare_my_load_prediction()
                        }
                        self.send_message(
                            msg=data,
                            to=candidate
                        )
                # Upon receiving handover success, the source remove the UE's record
                del self.candidates_record[ueid]
                del self.condition_record[ueid]

            # ================================================ Source
            elif task == RRC_RECONFIGURATION_COMPLETE:
                # no logic needs to be handled here
                assert (True)
                # ================================================ Target
            elif task == SN_STATUS_TRANSFER:
                data = {
                    "task": PATH_SWITCH_REQUEST,
                    "sourceid": data['from'],
                }
                self.send_message(
                    msg=data,
                    to=self.AMF
                )
            # ================================================ Candidate
            elif task == HANDOVER_CANCEL:
                ueid = data['ueid']
                expected_access_time, expected_leaving_time = self.estimated_access_handover_precise_time(ueid)
                # upon receving handover cancel, the candidate remove the UE's record
                del self.takeover_condition_record[ueid]
                self.access_Q.release_resource(ueid)
                self.decrease_my_load_potential(expected_access_time,
                                                SOURCE_HANDOVER_REQUEST_SIGNALLING_COUNT_ON_CANDIDATE)
                self.decrease_my_load_potential(expected_leaving_time, UE_HANDOVER_SIGNALLING_COUNT_ON_SOURCE)
            elif task == PATH_SWITCH_REQUEST_ACK:
                source = self.satellites[data['sourceid']]
                data = {
                    "task": UE_CONTEXT_RELEASE,
                    'priority_load': self.prepare_my_load_prediction()
                }
                self.send_message(
                    msg=data,
                    to=source
                )
            elif task == UE_CONTEXT_RELEASE:
                assert (True)
            else:
                assert False

    def estimated_access_handover_precise_time(self, ueid):
        serving_time = self.takeover_condition_record[ueid].ue_utility
        issue_time, expected_access_time = self.access_Q.return_expected_issue_access_time(ueid)
        expected_leaving_time = serving_time + issue_time
        return expected_access_time, expected_leaving_time

    def prepare_condition(self, ueid, sourceid, candidates, utilities):
        delay = self.decide_delay(ueid, sourceid, candidates, utilities)
        ue_utility = utilities[candidates.index(self.identity)]
        # TODO Should we consider the case when access time cannot happen with handover at the same time?
        if self.env.now + delay < self.DURATION:
            assert (self.coverage_info[ueid, self.identity, self.env.now + delay] == 1)
        expected_leaving_time = ue_utility + self.env.now
        condition = Sat_condition(access_delay=delay, ueid=ueid, satid=self.identity, sourceid=sourceid,
                                  ue_utility=ue_utility,
                                  future_potential_load=self.predicted_my_load_potential[expected_leaving_time],
                                  future_real_load=self.predicted_my_load[expected_leaving_time])
        self.access_Q.insert(ueid, delay)
        self.record_max_delay = max(self.record_max_delay, delay)
        return condition

    # def extend_array(self, arry, length, padding_value):
    #     current_length = len(arry)
    #     assert (current_length <= length)
    #     if current_length < length:
    #         return np.pad(arry, (0, length - current_length), 'constant', constant_values=(padding_value,))
    #     else:
    #         return arry

    def extend_array(self, arry, length):
        current_length = len(arry)
        assert current_length <= length, "The desired length must be greater than or equal to the current length."
        if current_length == 0:
            return np.zeros(length)
        elif current_length < length:
            # Generate padding values based on the distribution of the existing array values
            padding_length = length - current_length
            padding_values = np.random.choice(arry, size=padding_length)
            extended_array = np.concatenate((arry, padding_values))
            return extended_array
        else:
            return arry

    def candidates_selection(self, candidates, utilities):
        zipped_lists = list(zip(candidates, utilities))
        if SOURCE_SELECTION_ALG == SOURCE_SELECTION_RANDOM:
            random_selected_pairs = random.sample(zipped_lists, NUMBER_CANDIDATE)
            selected_candidates, selected_utilities = zip(*random_selected_pairs)
        elif SOURCE_SELECTION_ALG == SOURCE_SELECTION_LONGEST:
            sorted_zipped_lists = sorted(zipped_lists, key=lambda x: x[1], reverse=True)
            sorted_candidates, sorted_utilities = zip(*sorted_zipped_lists)
            selected_candidates = sorted_candidates[:NUMBER_CANDIDATE]
            selected_utilities = sorted_utilities[:NUMBER_CANDIDATE]
        elif SOURCE_SELECTION_ALG == SOURCE_SELECTION_OUR:
            pass
        return selected_candidates, selected_utilities

    def decide_delay(self, ueid, sourceid, candidates, utilities):
        assert (self.access_Q.counter - self.access_Q.max_access_slots == self.env.now)
        available_slots = self.access_Q.available_slots()
        # if CANDIDATE_ALG == CANDIDATE_ALG_EARLIEST:
        #     # greedy
        #     delay = min(available_slots) + 1
        # if CANDIDATE_ALG == CANDIDATE_ALG_RANDOM:
        #     # random
        #     delay = random.choice(available_slots) + 1
        if CANDIDATE_ALG == CANDIDATE_OUR:
            available_slots = self.access_Q.available_slots()
            if True not in available_slots:
                assert (False)
            loads = []
            for candidate_id in candidates:
                if candidate_id == self.identity:
                    myload = self.prepare_my_load_prediction()
                    my_real_load = np.array(myload[1])[1:]
                    my_potential_load = np.array(myload[2])[1:]
                    myload = my_real_load + my_potential_load
                    loads.append(myload)
                else:
                    otherload = self.prepare_other_load_prediction(candidate_id)
                    other_real_load = np.array(otherload[1])[1:]
                    other_potential_load = np.array(otherload[2])[1:]
                    otherload = other_real_load + other_potential_load
                    otherload = self.extend_array(otherload, len(available_slots))
                    loads.append(otherload)
            loads = np.array(loads)
            valid_indices = np.where(available_slots)[0]
            A_valid = loads[:, valid_indices]
            max_values = np.max(A_valid, axis=0)
            min_index_in_max_values = np.argmin(max_values)
            delay = valid_indices[min_index_in_max_values] + 1
        return int(delay)

    def decide_best_target(self, ueid):
        conditions = self.condition_record[ueid]
        if self.oracle is not None:
            targetid = self.oracle.query_next_satellite(ueid, self.identity)
        if self.oracle is not None and targetid != -1:
            # TODO not tested
            found = False
            for condition in conditions:
                if targetid == condition['satid']:
                    found = True
                    corresponding_delay = condition['access_delay']
            if not found:
                raise AssertionError("The UE did not receive condition from oracle arranged target satellite")
            return targetid, corresponding_delay
        # This clause if for non-oracle
        else:
            if SOURCE_DECISION_ALG == SOURCE_DECISION_OUR:
                min_sum = min(
                    c['future_potential_real_load'][0] + c['future_potential_real_load'][1] for c in conditions)
                min_conditions = [c for c in conditions if
                                  c['future_potential_real_load'][0] + c['future_potential_real_load'][1] == min_sum]
                selected_condition = random.choice(min_conditions)
            targetid = selected_condition['satid']
            delay = selected_condition['access_delay']
            return targetid, delay
