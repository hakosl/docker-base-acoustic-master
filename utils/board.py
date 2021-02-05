import datetime
import os
class Board():
    def __init__(self, log_path="/acoustic/log"):
        now = datetime.datetime.now().strftime("%Y-%d-%B--%H-%M-%S")
        self.path = f"{log_path}/{now}"
        os.makedirs(self.path)
