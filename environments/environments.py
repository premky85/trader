from cProfile import label
from environments.dataloaders.binance_dataloader import BinanceDataLoader
import numpy as np
from tqdm import tqdm
from tf_agents.environments import py_environment, utils, tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from sklearn import preprocessing


import matplotlib.pyplot as plt


class TFTraderEnv(py_environment.PyEnvironment):
    def __init__(
        self,
        path,
        sample_size: int = 64,
        train_steps: int = 1000,
        validation=False,
        discrete=True,
    ):

        self.discrete = discrete
        self.discrete_actions = {0: -0.5, 1: 0, 2: 0.5}
        max_action = 1
        if discrete:
            max_action = 2

        self.loader = BinanceDataLoader(
            path, sample_size=sample_size, train_steps=train_steps, overfit=True
        )

        self.sample_sizes = self.loader.get_sample_sizes()

        action_type = np.int32 if discrete else np.float32

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=action_type, minimum=0, maximum=max_action, name="action"
        )
        self._observation_spec = array_spec.ArraySpec(
            shape=(sum(self.sample_sizes) + 2,),
            dtype=np.float32,
            name="observation",
        )

        self.sample_size = sample_size
        self.steps = self.loader.get_steps()
        self.path = path
        self.validation = validation

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):

        self._clear_data()

        data = next(self.data_iterator)

        self.price_init = data["price"]

        balance_state = np.array([self.balance_fiat, self.balance_crypto])

        self._state = np.concatenate([balance_state, data["observation"]]).astype(
            np.float32
        )

        return ts.restart(self._state)

    def _step(self, action) -> ts.TimeStep:
        if self.discrete:
            action = self.discrete_actions[int(action)]
        else:
            action = (action - 0.5) * 2

        data = next(self.data_iterator)

        done = data["done"]
        self.i += 1
        self.trading_summary["price"].append(1000 * data["price_old"] / self.price_init)

        max_reward = (
            self.balance_net
            / data["price_old"]
            * abs(data["price"] - data["price_old"])
        )

        old_net = self.balance_net

        # buy or hold
        if action >= 0:
            transaction_amount = action * self.balance_fiat
            self.balance_fiat -= transaction_amount
            self.balance_crypto += transaction_amount / data["price"]

            if self.balance_fiat < 0.01:
                self.balance_fiat = 0

        # sell
        else:
            transaction_amount = action * self.balance_crypto
            self.balance_crypto += transaction_amount
            self.balance_fiat -= transaction_amount * data["price"]

            if self.balance_crypto < 0.0001:
                self.balance_crypto = 0

        self.fiat_balances.append(self.balance_fiat)
        self.crypto_balances.append(self.balance_crypto)

        new_net = self.balance_fiat + self.balance_crypto * data["price"]
        reward = new_net - old_net  # - max_reward

        if not self.validation:
            reward = reward / max_reward if reward > 1 else -0.1
            # if reward < -0.1:
            #     reward = -1
            # elif reward > 0.1:
            #     reward = 1
            # elif transaction_amount <= 0.00001:
            #     reward = -0.2

        # if transaction_amount <= 0.00001:
        #     reward = -0.1

        done = done or new_net < -100

        self.balance_net = new_net

        if self.validation:
            self.actions.append(action)
            self.rewards.append(reward)
            self.buy_and_hold.append((data["price"] / data["price_old"] - 1) * 1000)

            self.nets.append(self.balance_net)
            self.buy_and_hold_nets.append((data["price"] / self.price_init) * 1000)
            self.fiat_nets.append(self.balance_fiat)
            self.crypto_nets.append(self.balance_crypto * data["price"])

        balance_fiat_n = preprocessing.scale(self.fiat_balances)[-1]
        balance_crypto_n = preprocessing.scale(self.crypto_balances)[-1]

        balance_state = np.array([balance_fiat_n, balance_crypto_n])

        self._state = np.concatenate([balance_state, data["observation"]]).astype(
            np.float32
        )

        if done:
            if self.validation:
                self._describe_actions()

            # self._clear_data()
            term = ts.termination(self._state, reward)
            self._reset()
            return term

        return ts.transition(self._state, reward)

    def _clear_data(self):
        self.trading_summary = {"bank": [], "USD": [], "Profit": [], "price": []}

        self.data_iterator = self.loader.process_data(validation=self.validation)

        self.action_regularizer = 0.8

        self.balance_fiat = 1000  # money balance
        self.balance_crypto = 0  # crypto balance
        self.balance_net = 1000  # net worth

        self.fiat_balances = [self.balance_fiat]
        self.crypto_balances = [self.balance_crypto]

        self.i = 0

        self.reward = 0
        self.step_reward = 0
        self.actions = []
        self.rewards = []
        self.buy_and_hold = []

        self.nets = [self.balance_fiat]
        self.fiat_nets = [self.balance_fiat]
        self.crypto_nets = [0]
        self.buy_and_hold_nets = [self.balance_fiat]

    def _describe_actions(
        self,
    ):
        a = np.array(self.actions)
        r = np.array(self.rewards)
        b_h = np.array(self.buy_and_hold)

        n = np.array(self.nets)
        b_h_n = np.array(self.buy_and_hold_nets)
        f_n = np.array(self.fiat_nets)
        c_n = np.array(self.crypto_nets)
        reward_output = " | Rewards - mean: {:.3f}, STD: {:.3f}, min: {:.3f}, max: {:.3f}, total: {:.3f}, B&H: {:.3f}".format(
            np.mean(r), np.std(r), np.min(r), np.max(r), np.sum(r), np.sum(b_h)
        )
        if self.discrete:
            unique, counts = np.unique(a, return_counts=True)

            tqdm.write(
                "Env. reset | Actions - mean: {:.3f}, distribution: {}{}".format(
                    np.mean(a),
                    dict(zip(unique, counts)),
                    reward_output if self.validation else "",
                )
            )
        else:
            tqdm.write(
                "Env. reset | Actions - mean: {:.3f}, STD: {:.2f}, min: {:.2f}, max: {:.2f}{}".format(
                    np.mean(a),
                    np.std(a),
                    np.min(a),
                    np.max(a),
                    reward_output if self.validation else "",
                )
            )

        plt.plot(r, label="Agent")
        plt.plot(b_h, label="Buy & Hold")
        plt.title("Reward")
        plt.legend()
        plt.savefig(
            "plots/validation_reward/{:.2f}_{:.2f}.png".format(np.sum(r), np.sum(b_h))
        )
        plt.clf()

        plt.plot(n, label="Agent")
        plt.plot(b_h_n, label="Buy & Hold")
        plt.plot(c_n, label="Crypto balance", color="green")
        plt.plot(f_n, label="Fiat balance", color="red")
        plt.title("Net Worth")
        plt.legend()
        plt.savefig("plots/net_worth/{:.2f}_{:.2f}.png".format(n[-1], b_h_n[-1]))
        plt.clf()

        plt.close()


if __name__ == "__main__":
    train_steps = 100
    env = TFTraderEnv("data/Binance_BTCUSDT_minute.csv", train_steps=train_steps)
    utils.validate_py_environment(env, episodes=100)

    # env = tf_py_environment.TFPyEnvironment(env)
