import enum
import struct


class SpeedState(enum.Enum):
    Normal_GO = 1
    Limit40_GO = 2


class Traffic:
    def __init__(self, connection):
        self.connection = connection
        self.state = {'red_light': False, "speed": SpeedState.Normal_GO, "stop:": False}

    def red_light_action(self):
        self.state["speed"] = SpeedState.Normal_GO
        self.state["red_light"] = True
        self.connection.write(struct.pack('<L', 4))
        self.connection.write(b"stop")
        self.connection.flush()
        print("red")

    def green_light_action(self):
        self.state["red_light"] = False
        self.state["stop"] = False
        self.connection.write(struct.pack('<L', len(self.state["speed"].name)))
        self.connection.write(bytes(self.state["speed"].name.encode("utf-8")))
        self.connection.flush()

    def limit40_action(self):
        self.state["speed"] = SpeedState.Limit40_GO
        if self.state["red_light"]:
            self.connection.write(struct.pack('<L', 4))
            self.connection.write(b"stop")
            self.connection.flush()
        else:
            self.connection.write(struct.pack('<L', len(self.state["speed"].name)))
            self.connection.write(bytes(self.state["speed"].name.encode("utf-8")))
            self.connection.flush()

    def end40limit_action(self):
        self.state["speed"] = SpeedState.Normal_GO
        if self.state["red_light"]:
            self.connection.write(struct.pack('<L', 4))
            self.connection.write(b"stop")
            self.connection.flush()
        else:
            self.connection.write(struct.pack('<L', len(self.state["speed"].name)))
            self.connection.write(bytes(self.state["speed"].name.encode("utf-8")))
            self.connection.flush()

    def stop_sign_action(self):
        self.connection.write(struct.pack('<L', 4))
        self.connection.write(b"stop")
        self.connection.flush()

    def person_action(self):
        self.state["stop"] = True
        self.connection.write(struct.pack('<L', 4))
        self.connection.write(b"stop")
        self.connection.flush()

    def car_action(self):
        self.state["stop"] = True
        self.connection.write(struct.pack('<L', 4))
        self.connection.write(b"stop")
        self.connection.flush()

    def left_action(self):
        self.state["stop"] = False
        if self.state["red_light"]:
            self.connection.write(struct.pack('<L', 4))
            self.connection.write(b"stop")
            self.connection.flush()
        else:
            self.connection.write(struct.pack('<L', 4))
            self.connection.write(b"left")
            self.connection.flush()

    def right_action(self):
        self.state["stop"] = False
        if self.state["red_light"]:
            self.connection.write(struct.pack('<L', 4))
            self.connection.write(b"stop")
            self.connection.flush()
        else:
            self.connection.write(struct.pack('<L', 5))
            self.connection.write(b"right")
            self.connection.flush()

