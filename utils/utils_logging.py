
import os
import time
import logging

class Logger():
    def __init__(self, model_name, log_path='./logs', multi=False):
        self.model_name = model_name
        self.log_path = log_path
        if os.path.exists(self.log_path) is False:
            os.mkdir(self.log_path)

        # Defalut Logger(1 and 2)
        self.train = self._init_logger(logging.getLogger(f'{self.model_name}-train')) # Defalut
        self.val = self._init_logger(logging.getLogger(f'{self.model_name}-val')) # Default
        # Extra Logger
        if multi:
            self.center = self._init_logger(logging.getLogger(f'{self.model_name}-center'))
            self.edge = self._init_logger(logging.getLogger(f'{self.model_name}-edge') )
        
    def _init_logger(self, logger):
        logger.setLevel(logging.INFO)
        
        rq = time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time()))
        log_name = os.path.join(self.log_path, f'{rq}-{logger.name}.log')
        fh = logging.FileHandler(log_name, mode='w')
        fh.setLevel(logging.INFO)
        
        formatter = logging.Formatter("%(asctime)s - %(filename)s [line:%(lineno)d] - %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        return logger