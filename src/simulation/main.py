import sys
from Load import *
from UE import *
from Satellite import *
from AMF import *
from Oracle import *
import time
import os


def initial_assignment(UEss, Satss, oracle):
    C = UEss[0].coverage_info
    if oracle is None:
        # random assign
        for ueid in UEss:
            ue = UEss[ueid]
            possible_satellites = np.where(C[ueid, :, 0] == 1)[0]
            assert (len(possible_satellites) > 0)
            satid = random.choice(possible_satellites)
            ue.serving_satellite = Satss[satid]
            ue.serving_satellite_history.append(satid)
            serving_length = ue.estimate_serving_length(satid)
            Satss[satid].increment_my_load(serving_length, UE_HANDOVER_SIGNALLING_COUNT_ON_SOURCE)
    else:
        for ueid in UEss:
            ue = UEss[ueid]
            satid = oracle.query_init_satellite(ueid)
            assert (C[ueid, satid, 0] == 1)
            ue.serving_satellite = Satss[satid]
            ue.serving_satellite_history.append(satid)
            serving_length = ue.estimate_serving_length(satid)
            Satss[satid].increment_my_load(serving_length, UE_HANDOVER_SIGNALLING_COUNT_ON_SOURCE)


def monitor_timestamp(env):
    while True:
        if env.now % 100 == 0:
            print(f"Current simulation time {env.now}", file=sys.stderr)
        yield env.timeout(1)


if __name__ == "__main__":
    # configuration
    # When oracle_simulation is False, the oracle_assignment could be True/False
    # When oracle_simulation is True, the oracle_assignment must be True
    if oracle_simulation:
        assert (oracle_assignment)
    random.seed(10)

    # Loading scenarios
    beginning = time.time()
    UEs_template, satellites_template, C = load_scenario()
    DURATION = C.shape[2]
    env = simpy.Environment()

    assignment_oracle = None
    simulation_oracle = None
    if oracle_assignment:
        assignment_oracle = Oracle()
    if oracle_simulation:
        simulation_oracle = assignment_oracle

    UEs = {}
    for ue_template in UEs_template:
        ID = ue_template.ID
        if ID not in UEs:
            UEs[ID] = UE(
                identity=ID,
                position_x=ue_template.x,
                position_y=ue_template.y,
                coverage_info=C,
                oracle=simulation_oracle,
                env=env
            )
        else:
            print("ERROR")

    amf = AMF(coverage_info=C, env=env)
    Satellites = {}
    for sat_template in satellites_template:
        ID = sat_template.ID
        if ID not in Satellites:
            Satellites[ID] = Satellite(
                identity=ID,
                position_x=sat_template.x,
                position_y=sat_template.y,
                height=sat_template.h,
                coverage_r=sat_template.r,
                velocity=sat_template.v,
                sind=sat_template.sind,
                cosd=sat_template.cosd,
                coverage_info=C,
                max_access_opportunity=max_access_opportunity,
                max_access_slots=max_access_slots,
                env=env,
                oracle=simulation_oracle,
            )
            Satellites[ID].AMF = amf
        else:
            print("ERROR")

    amf.satellites = Satellites

    for ueid in UEs:
        UEs[ueid].satellites = Satellites

    for satid in Satellites:
        Satellites[satid].UEs = UEs
        Satellites[satid].satellites = Satellites

    print(f"Loading scenario in the simulation takes {time.time() - beginning}")
    # ========= UE, SAT objectives are ready =========
    initial_assignment(UEs, Satellites, assignment_oracle)

    env.process(monitor_timestamp(env))

    print('============= Experiment Log =============')
    env.run(until=DURATION)
    print('============= Experiment Ends =============')
    counter = allCounters(Satellites, UEs)
    counter.give_result()
