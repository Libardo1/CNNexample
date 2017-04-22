class Config():
    """
    Class to hold all model hyperparams.
    """
    def __init__(self,
                 batch_size=140,
                 patch_size=5,
                 image_size=28,
                 num_labels=10,
                 num_channels=1,
                 num_filters_1=16,
                 num_filters_2=32,
                 hidden_nodes_1=60,
                 hidden_nodes_2=40,
                 hidden_nodes_3=20,
                 learning_rate=0.9,
                 steps_for_decay=100,
                 decay_rate=0.96):
        """
        :type batch_size: int
        :type patch_size: int
        :type image_size: int
        :type num_channels: int
        :type num_filters_1: int
        :type num_filters_2: int
        :type hidden_nodes_1: int
        :type hidden_nodes_2: int
        :type hidden_nodes_3: int
        :type learning_rate: float
        :type steps_for_decay: float
        :type decay_rate: float
        """
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_labels = num_labels
        self.num_channels = num_channels
        self.num_filters_1 = num_filters_1
        self.num_filters_2 = num_filters_2
        self.hidden_nodes_1 = hidden_nodes_1
        self.hidden_nodes_2 = hidden_nodes_2
        self.hidden_nodes_3 = hidden_nodes_3
        self.learning_rate = learning_rate
        self.steps_for_decay = steps_for_decay
        self.decay_rate = decay_rate