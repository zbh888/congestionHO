import json



class Base:
    def __init__(self,
                 identity,
                 position_x,
                 position_y,
                 object_type,
                 env
                 ):
        self.type = object_type
        self.identity = identity
        self.position_x = position_x
        self.position_y = position_y
        self.env = env
        self.type = object_type

    def init(self):
        print(
            f"{self.type} {self.identity} deployed at time {self.env.now}, positioned at ({self.position_x},{self.position_y})")
        if self.type == "UE" and self.serving_satellite is not None:
            print(
                f"{self.type} {self.identity} is served by {self.serving_satellite.type} {self.serving_satellite.identity}")
        yield self.env.timeout(1)

    def send_message(self, msg, to):
        """ Send the message with delay simulation

        Args:
            delay: The message propagation delay
            msg: the json object needs to be sent
            Q: the Q of the receiver
            to: the receiver object

        """
        msg['from'] = self.identity
        msg['to'] = to.identity
        msg = json.dumps(msg)
        print(f"{self.type} {self.identity} sends {to.type} {to.identity} the message {msg} at {self.env.now}")
        to.messageQ.put(msg)
