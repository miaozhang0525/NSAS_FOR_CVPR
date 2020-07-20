class BaseOptions():
    def initialize(self):
        self.dataroot = '../' # path to the dir of the dataset
        self.name = 'Default' # Name of the experiment

class TrainOptions(BaseOptions):
    def __init__(self):

        BaseOptions.initialize(self)
        self.initial_temp = 0.5 # innitial softmax temperature
        self.anneal_rate = 0.1 # annealation rate of softmax temperature
