import torch


class MarioConfig:
    def __init__(self):
        # hyper config
        self.max_num_gpus = 1
        self.num_workers = 32
        self.discount = 0.999

        self.observation_space = (84, 84, 3)
        self.action_space = 256 + 20 + 8
        import os
        import datetime
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../results",
                                         os.path.basename(__file__)[:-3], datetime.datetime.now().strftime(
                "%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.save_log = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = int(100 * 1e6)  # Total number of training steps (ie weights update according to a batch)

        # Alg config
        self.lambda_ = 0.95

        # Actor config

        # Learner config
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available
        self.batch_size = 32  # Number of parts of games to train on at each training step
        self.checkpoint_interval = int(8)  # Number of training steps before using the model for self-playing

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD
        self.cofentropy = 1e-3
        self.v_scaling = 0.5
        self.clip_param = 0.15
        self.lr_init = 5e-4  # Initial learning rate
        self.replay_buffer_size = int(1e3)  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 16  # Number of game moves to keep for every batch element
