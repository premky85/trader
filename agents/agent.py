import tensorflow as tf
import numpy as np
from tf_agents.policies import (
    py_tf_eager_policy,
    policy_saver,
    policy_loader,
    greedy_policy,
)
from tf_agents.environments import tf_py_environment
from tf_agents.utils import common
from tf_agents.drivers import py_driver
from tqdm import tqdm
import os
from statistics import mean
from environments.environments import TFTraderEnv


class Agent:
    def __init__(self, sample_size, batch_size, discrete) -> None:
        self.sample_size = sample_size
        self.env = TFTraderEnv(
            train_steps=100,
            path="data/Binance_BTCUSDT_1h.csv",
            sample_size=sample_size,
            discrete=discrete,
        )
        self.sample_sizes = self.env.sample_sizes
        self.env = tf_py_environment.TFPyEnvironment(self.env)
        self.n_states = self.env.time_step_spec().observation.shape[0]

        self.validation_env = TFTraderEnv(
            path="data/Binance_BTCUSDT_1h.csv",
            train_steps=100,
            validation=True,
            sample_size=sample_size,
            discrete=discrete,
        )
        self.validation_env = tf_py_environment.TFPyEnvironment(self.validation_env)

        self.batch_size = batch_size
        self.learning_rate_i = 0
        self.global_step = tf.compat.v1.train.get_or_create_global_step()
        self.lr_counter_threshold = 5
        self.reset_lr_threshold = False
        self.discrete = discrete

    def train(self, episodes=5000, validation_freq=100, validation_runs=10):
        agent = self.agent
        env = self.env
        buffer = self.buffer
        global_step = self.global_step

        agent.train = common.function(agent.train)
        agent.train_step_counter.assign(0)

        time_step = env.reset()

        policy = greedy_policy.GreedyPolicy(agent.policy)

        train_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(
                os.getcwd(), "checkpoints", "train_{}".format(self.name)
            ),
            max_to_keep=1,
            agent=agent,
            policy=agent.policy,
            buffer=buffer,
            global_step=global_step,
        )

        train_checkpointer.initialize_or_restore()
        tf_policy_saver = policy_saver.PolicySaver(agent.policy)

        collect_driver = py_driver.PyDriver(
            env,
            py_tf_eager_policy.PyTFEagerPolicy(
                agent.collect_policy, use_tf_function=True, batch_time_steps=False
            ),
            self.observers,
            max_steps=10,
        )

        losses = []

        best_loss = float("inf")

        loss_delay_counter = 0

        for ep in tqdm(range(episodes)):

            time_step, _ = collect_driver.run(time_step)

            # Sample a batch of data from the buffer and update the agent's network.
            iterator = iter(self.dataset)
            experience, unused_info = next(iterator)
            train_loss = agent.train(experience).loss.numpy()
            losses.append(train_loss)

            if ep % validation_freq == 0 and ep > 0:

                reward = self.compute_reward(num_episodes=validation_runs)
                output = "Episode: {0}, Validation average reward: {1}, Avg. train loss: {2}, Loss delay counter {3}".format(
                    ep // validation_freq,
                    reward.numpy()[0],
                    mean(losses),
                    loss_delay_counter,
                )
                tqdm.write(output)

                if mean(losses) < best_loss * 0.8:
                    best_loss = mean(losses) * 1.2
                    loss_delay_counter = 0
                    train_checkpointer.save(global_step)

                elif (
                    loss_delay_counter >= self.lr_counter_threshold
                    and self.learning_rate_i < len(self.learning_rates) - 1
                ):
                    if self.reset_lr_threshold:
                        best_loss = mean(losses)

                    train_checkpointer.initialize_or_restore()
                    self._get_dataset(buffer)

                    self.learning_rate_i += 1
                    self._update_lr()

                    output = "Reducing LR to {}".format(
                        self.learning_rates[self.learning_rate_i]
                    )
                    tqdm.write(output)
                    loss_delay_counter = 0

                elif self.learning_rate_i >= len(self.learning_rates):
                    self.learning_rate_i = 0

                else:
                    loss_delay_counter += 1

                losses = []

    def compute_reward(self, num_episodes=10):
        total_reward = 0.0
        for ep in range(num_episodes):
            time_step = self.validation_env.reset()
            while not time_step.is_last():
                action_step = self.agent.policy.action(time_step)
                time_step = self.validation_env.step(action_step.action)
                total_reward += time_step.reward

        return total_reward / num_episodes

    def _get_dataset(self, buffer):
        self.dataset = buffer.as_dataset(
            num_parallel_calls=3, sample_batch_size=self.batch_size, num_steps=2
        ).prefetch(3)
