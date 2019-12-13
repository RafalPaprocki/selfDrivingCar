import time
import io
import threading
import picamera
from threading import Condition


class StreamingOutput(object):
    def __init__(self):
        self.frame = None
        self.buffer = io.BytesIO()
        self.condition = Condition()
        self.size = None
    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            self.size = self.buffer.tell()
            self.buffer.seek(0)
            self.frame = self.buffer.read(self.size)
            self.buffer.seek(0)
        return self.buffer.write(buf)


class Camera(object):
    thread = None
    frame = None
    output = StreamingOutput()

    def initialize(self):
        if Camera.thread is None:
            Camera.thread = threading.Thread(target=self._thread)
            Camera.thread.start()
            time.sleep(1.5)

    @classmethod
    def _thread(cls):
        with picamera.PiCamera(resolution='640x380', framerate=24) as camera:
            Camera.output = StreamingOutput()
            camera.start_recording(cls.output, format='mjpeg')
            time.sleep(100000)

    def take_frame(self):
        return Camera.output.frame
