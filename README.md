# SelfDrivingCar
This project consist of two application:
1) Client application for raspberry pi.
2) Server application which can run for instance on personal computer.

### Client application
This application run on raspberry pi, and it grab data from sensors, and frames from camera. This application 
communicate with server and send this data 
to server. As a response, server send acction (run, stop, turn right etc) that should be taken by car.

### Server application
Application receive frames and sensors data from client. Then it use pretrained model to detect objects (such as road signs, traffic lights, cars or person).
Based on data detected from picture and sensors application determines action, and send it back to client.
