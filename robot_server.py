import zmq
import pickle
import robotic as ry


class RobotServer:

    def __init__(self, address: str="tcp://*:1234", on_real: bool=False, verbose: int=0):
        
        self.C = ry.Config()
        self.C.addFile(ry.raiPath("../rai-robotModels/scenarios/pandasTable.g"))
        self.C.view(False)
        self.bot = ry.BotOp(self.C, on_real)
        self.bot.home(self.C)
        self.verbose = verbose

        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        self.address = address
        self.socket.bind(address)

    def send_error_message(self, text):
        message = {}
        message["success"] = False
        message["text"] = text
        to_send = pickle.dumps(message)
        self.socket.send(to_send)

    def execute_command(self, message: dict):

        if message["command"] == "move":
        
            self.bot.move(message["path"], message["times"])
            while self.bot.getTimeToEnd() > 0:
                self.bot.sync(self.C)

        elif message["command"] == "moveAutoTimed":
        
            self.bot.moveAutoTimed(message["path"], message["time_cost"])
            while self.bot.getTimeToEnd() > 0:
                self.bot.sync(self.C)

        elif message["command"] == "home":
            self.bot.home(self.C)
        
        else:
            raise Exception(f"Command {message['command']} not implemented.")
        
    def run(self):

        if self.verbose:
            print("Started server at ", self.address)

        running = True
        while running:
            client_input = self.socket.recv()
            try:
                client_input = pickle.loads(client_input)
            except Exception as e:
                self.send_error_message(f"Error while loading message: {e}")
                
            if self.verbose:
                print(f"Received request: {client_input}")

            try:
                if client_input["command"] == "close":
                    running = False
                else:
                    self.execute_command(client_input)

            except Exception as e:
                self.send_error_message(f"Error while executing command: {e}")
            
            message = {}
            message["success"] = True
            message["command"] = client_input["command"]
            to_send = pickle.dumps(message)
            self.socket.send(to_send)

            if self.verbose:
                print("Sent a response.")


if __name__ == "__main__":
    robot = RobotServer(verbose=1)
    robot.run()
