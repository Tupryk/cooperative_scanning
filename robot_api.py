import zmq
import pickle
import numpy as np


class RobotAPI:

    def __init__(self,
                 address: str="tcp://localhost:1234",
                 verbose: int=0):
        
        self.verbose = verbose
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(address)

    def send_message(self, message: dict) -> dict:
        if self.verbose:
            print(f"Sending mesage: {message}...")
        
        to_send = pickle.dumps(message)
        self.socket.send(to_send)

        reply = self.socket.recv()
        reply = pickle.loads(reply)
        
        if self.verbose:
            print(f"Received reply: {reply}")
        
        return reply

    def move(self, path: np.ndarray, times: list[float]) -> bool:
        
        message = {}
        message["command"] = "move"
        message["path"] = path
        message["times"] = times

        reply = self.send_message(message)
        
        success = reply["success"]
        return success
    
    def moveAutoTimed(self, path: np.ndarray, time_cost: float) -> bool:
        
        message = {}
        message["command"] = "moveAutoTimed"
        message["path"] = path
        message["time_cost"] = time_cost

        reply = self.send_message(message)
        
        success = reply["success"]
        return success
    
    def home(self) -> bool:
        
        message = {}
        message["command"] = "home"

        reply = self.send_message(message)
        
        success = reply["success"]
        return success
    
    def close(self) -> bool:
        
        message = {}
        message["command"] = "close"

        reply = self.send_message(message)
        
        success = reply["success"]
        return success


if __name__ == "__main__":
    robot_api = RobotAPI(verbose=1)
    robot_api.home()
    robot_api.close()
