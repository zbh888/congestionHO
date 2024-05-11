import sys
from Load import *
from UE import *
from Satellite import *
from Oracle import *
import time

def initial_assignment(UEss, Satss, assignment):
    C = UEss[0].coverage_info
    if assignment is None:
        for ueid in UEss:
            ue = UEss[ueid]
            possible_satellites = np.where(C[ueid, :, 0] == 1)[0]
            assert (len(possible_satellites) > 0)
            satid = random.choice(possible_satellites)
            ue.serving_satellite = Satss[satid]


def monitor_timestamp(env):
    while True:
        if env.now % 100 == 0:
            print(f"Current simulation time {env.now}", file=sys.stderr)
        yield env.timeout(1)


if __name__ == "__main__":
    # configuration
    oracle_mode = True
    random.seed(10)

    # Loading scenarios
    beginning = time.time()
    UEs_template, satellites_template, C = load_scenario()
    DURATION = C.shape[2]
    env = simpy.Environment()

    oracle = None
    if oracle_mode:
        oracle = Oracle()
        
    UEs = {}
    for ue_template in UEs_template:
        ID = ue_template.ID
        if ID not in UEs:
            UEs[ID] = UE(
                identity=ID,
                position_x=ue_template.x,
                position_y=ue_template.y,
                coverage_info=C,
                oracle = oracle,
                env=env
            )
        else:
            print("ERROR")

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
                env=env
            )
        else:
            print("ERROR")

    for ueid in UEs:
        UEs[ueid].satellites = Satellites

    for satid in Satellites:
        Satellites[satid].UEs = UEs
        Satellites[satid].satellites = Satellites    
        
    print(f"Loading scenario in the simulation takes {time.time() - beginning}")
    # ========= UE, SAT objectives are ready =========
    initial_assignment(UEs, Satellites, None)

    env.process(monitor_timestamp(env))

    print('============= Experiment Log =============')
    env.run(until=DURATION)
    print('============= Experiment Ends =============')
    counter = allCounters(Satellites)
    # counter.generate_heap_map(100)
