import enum
import struct
from commands import Commands


class SpeedState(enum.Enum):
    Normal_GO = 1
    Limit40_GO = 2


class Traffic:
    def __init__(self):
        self.state = {'red_light': False, "speed": SpeedState.Normal_GO, "stop:": False}

    def red_light_action(self):
        self.state["stop"] = True
        self.state["red_light"] = True
        return Commands.Stop

    def green_light_action(self):
        self.state["red_light"] = False
        self.state["stop"] = False
        return Commands[self.state["speed"].name]

    def limit40_action(self):
        self.state["speed"] = SpeedState.Limit40_GO
        if self.state["red_light"]:
            return Commands.Stop
        else:
            return Commands[self.state["speed"].name]

    def end40limit_action(self):
        self.state["speed"] = SpeedState.Normal_GO
        if self.state["red_light"]:
            return Commands.Stop
        else:
            return Commands[self.state["speed"].name]

    def stop_sign_action(self):
        self.state["stop"] = True
        return Commands.Stop

    def person_action(self):
        self.state["stop"] = True
        return Commands.Stop

    def car_action(self):
        self.state["stop"] = True
        return Commands.Stop

    def left_action(self):
        self.state["stop"] = False
        if self.state["red_light"]:
            return Commands.Stop
        else:
            return Commands.Left

    def right_action(self):
        self.state["stop"] = False
        if self.state["red_light"]:
            return Commands.Stop
        else:
            return Commands.Right

    def none_traffic_object_action(self, distance):
        if self.state["red_light"]:
            return Commands.Stop
        else:
            if distance < 11:
                return Commands.Withdraw
            else:
                return Commands[self.state["speed"].name]