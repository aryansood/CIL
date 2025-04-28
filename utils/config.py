import pprint

class Configuration(object):
    """Configuration parameters exposed via the commandline."""

    def __init__(self, adict):
        self.__dict__.update(adict)

    def __str__(self):
        return pprint.pformat(vars(self), indent=4)

    @staticmethod
    def parse_cmd():
        parser = argparse.ArgumentParser()

        # Model.
        parser.add_argument('--data_workers', type=int, default=2, help='Number of parallel threads for data loading.')
        parser.add_argument('--print_every', type=int, default=400, help='Print stats to console every so many iters.')
        parser.add_argument('--eval_every', type=int, default=400, help='Evaluate validation set every so many iters.')
        parser.add_argument('--tag', default='', help='A custom tag for this experiment.')
        parser.add_argument('--seed', type=int, default=42, help='Random number generator seed.')

        # Data.
        parser.add_argument('--seed_seq_len', type=int, default=120, help='Number of frames for the seed length.')
        parser.add_argument('--target_seq_len', type=int, default=24, help='How many frames to predict.')

        # Learning configurations.
        parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate.')
        parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs.')
        parser.add_argument('--bs_train', type=int, default=16, help='Batch size for the training set.')
        parser.add_argument('--bs_eval', type=int, default=16, help='Batch size for valid/test set.')

        # model config
        parser.add_argument('--n_history', type=int, default=48, help="Number of past frames used to predict the next one")

        config = parser.parse_args()
        return Configuration(vars(config))

    @staticmethod
    def from_json(json_path):
        """Load configurations from a JSON file."""
        with open(json_path, 'r') as f:
            config = json.load(f)
            return Configuration(config)

    def to_json(self, json_path):
        """Dump configurations to a JSON file."""
        with open(json_path, 'w') as f:
            s = json.dumps(vars(self), indent=2, sort_keys=True)
            f.write(s)


if __name__ == "__main__":
    # run the following command: `python config.py --`
    Configuration.parse_cmd()