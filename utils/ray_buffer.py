import torch
import copy
import numpy
import ray


@ray.remote
class RemoteReplayBuffer:
    """
    Class which run in a dedicated thread to store played games and generate batch.
    """

    def __init__(self, initial_checkpoint, initial_buffer, config):
        self.config = config
        self.buffer = copy.deepcopy(initial_buffer)
        self.num_played_games = initial_checkpoint["num_played_games"]
        self.num_played_steps = initial_checkpoint["num_played_steps"]
        self.total_samples = sum([len(game_history) for game_history in self.buffer.values()])

        if self.total_samples > 0:
            print(
                f"Replay buffer initialized with {self.total_samples} samples ({self.num_played_games} games).\n"
            )

    def save_game(self, game_history, shared_storage=None):
        self.buffer[self.num_played_games] = game_history
        self.num_played_games += 1
        self.num_played_steps += len(game_history)
        self.total_samples += len(game_history)

        if self.config.replay_buffer_size < len(self.buffer):
            del_id = self.num_played_games - len(self.buffer)
            self.total_samples -= len(self.buffer[del_id])
            del self.buffer[del_id]

        if shared_storage:
            shared_storage.set_info.remote("num_played_games", self.num_played_games)
            shared_storage.set_info.remote("num_played_steps", self.num_played_steps)

    def get_buffer(self):
        return self.buffer

    def sample_game(self):
        """
        Sample game from buffer either uniformly or according to some priority.
        See paper appendix Training.
        """
        game_index = numpy.random.choice(len(self.buffer))
        game_id = self.num_played_games - len(self.buffer) + game_index
        return game_id, self.buffer[game_id]

    def sample_n_games(self, n_games):
        selected_games = numpy.random.choice(list(self.buffer.keys()), n_games)
        ret = [(game_id, self.buffer[game_id])
               for game_id in selected_games]
        return ret

    def sample_position(self, game_history):
        """
        Sample position from game either uniformly or according to some priority.
        See paper appendix Training.
        """
        position_index = numpy.random.choice(len(game_history.root_values))
        return position_index

    def update_game_history(self, game_id, game_history):
        # The element could have been removed since its selection and update
        if next(iter(self.buffer)) <= game_id:
            self.buffer[game_id] = game_history
