import simpy

from Base import *
from Config import *


class AMF(Base):
    def __init__(self,
                 coverage_info,
                 env):

        Base.__init__(self,
                      identity=-1,
                      position_x=0,
                      position_y=0,
                      coverage_info=coverage_info,
                      env=env,
                      object_type="AMF")

        # Config Initialization
        self.satellites = None

        # Running process
        self.env.process(self.init())  # Print Deployment information
        self.env.process(self.handle_messages())

    def handle_messages(self):
        """ Get the task from message Q and start a CPU processing process """
        while True:
            msg = yield self.messageQ.get()
            data = json.loads(msg)
            assert (data['to'] == self.identity)
            task = data['task']
            if task == PATH_SWITCH_REQUEST:
                satid = data['from']
                satellite = self.satellites[satid]
                data = {
                    "task": PATH_SWITCH_REQUEST_ACK,
                    "sourceid": data['sourceid']
                }
                self.send_message(
                    msg=data,
                    to=satellite
                )
            else:
                print(task)
                assert False
