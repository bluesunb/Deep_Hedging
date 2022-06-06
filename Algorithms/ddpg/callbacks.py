import time

from stable_baselines3.common.callbacks import BaseCallback


class ReportCallbacks(BaseCallback):
    def __init__(self, verbose: int= 0):
        super(ReportCallbacks, self).__init__(verbose=verbose)
        self.training_start = 0
        self.rollout_start = 0
        self.now_time = 0

    def _on_training_start(self) -> None:
        self.now_time = time.time()
        self.training_start = self.num_timesteps
        print(f'[Training Start]',)

    def _on_training_end(self) -> None:
        print(f'[Training End]  steps: {self.num_timesteps - self.training_start}'
              f'\ttimes: {time.time() - self.now_time}')

    # def _on_rollout_start(self) -> None:
    #     self.rollout_start = self.num_timesteps
    #
    # def _on_rollout_end(self) -> None:
    #     print(f'\t[Rollout End]: {self.num_timesteps - self.rollout_start}')

    def _on_step(self) -> bool:
        return True
