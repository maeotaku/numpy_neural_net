class ConfigEnum:
    XOR, IRIS, MNIST = range(3)

#factory-like class which produces different configs for each dataset tested
class hyperparams():

    def __init__(self, config_enum=1): #default is iris based on reqs
        if config_enum==ConfigEnum.XOR:
            import data.config_xor as config
        elif config_enum==ConfigEnum.IRIS:
            import data.config_iris as config
        else:
            import data.config_mnist as config
        self.config = config
        self.batch_size = config.batch_size
        self.validate_every_no_of_batches = config.validate_every_no_of_batches
        self.epochs = config.epochs
        self.input_size = config.input_size
        self.output_size = config.output_size
        self.hidden_shapes = config.hidden_shapes
        self.lr = config.lr
        self.has_dropout= config.has_dropout
        self.dropout_perc= config.dropout_perc
        self.output_log = config.output_log
        self.ds_train = config.ds_train
        self.ds_test = config.ds_test
        self.ds_val = config.ds_val

    def split_again(self, perc_train, perc_val):
        self.ds_train, self.ds_val, self.ds_test = self.config.splitter.split(self.batch_size, perc_train, perc_val)
