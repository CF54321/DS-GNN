import logging


class Logger(object):
    Initialized = False

    @staticmethod
    def initialize(path_to_log_file):
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(levelname)-8s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            handlers=[logging.FileHandler(path_to_log_file, mode='w'),
                                      logging.StreamHandler()])
        Logger.Initialized = True

    @staticmethod
    def log(level, message):
        assert Logger.Initialized, 'Logger has not been initialized'
        logging.log(level, message)

    @staticmethod
    def d(message):
        Logger.log(logging.DEBUG, message)

    @staticmethod
    def i(message):
        Logger.log(logging.INFO, message)

    @staticmethod
    def w(message):
        Logger.log(logging.WARNING, message)

    @staticmethod
    def e(message):
        Logger.log(logging.ERROR, message)


if __name__ == '__main__':
    log = Logger

    # 初始化
    log.initialize(r'test.log')

    # 输出
    log.i('Start:')
    for step in range(20):
        log.w('[Step {step}] Avg. Loss = , Learning Rate =  ( samples/sec; ETA  hrs)'.format(step=step))