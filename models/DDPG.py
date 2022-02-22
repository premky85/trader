from models.AgentNet import AgentNetActor, AgentNetCritic
from utils.helpers import Buffer, OUActionNoise
from environments.environment_continuous import TraderEnv
from environments.dataloaders.binance_dataloader import BinanceDataLoader
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class DDPG:
    def __init__(self, sample_size = 64, model_weights_id = None) -> None:
        self.sample_size = sample_size
        self.env = TraderEnv('./data/Binance_BTCUSDT_minute.csv', sample_size=sample_size, train_steps=5000)

        self.batch_size = 128
        self.n_actions = self.env.action_space.shape[0]
        self.n_states = self.env.observation_space.shape[0]
        self.min_action = self.env.action_space.low[0]
        self.max_action = self.env.action_space.high[0]

        self.__get_models(model_weights_id)

        # Learning rates for actor-critic models
        self.learning_rates = [(1e-3, 2e-3), (5e-4, 1e-3), (1e-4, 5e-4), (5e-5, 1e-4), (1e-5, 5e-5), (1e-6, 5e-6), (5e-7, 5e-7), (1e-7, 1e-7)]
        self.learning_rate_i = 1 # 3

        self.critic_optimizer = tf.keras.optimizers.Adam(self.learning_rates[self.learning_rate_i][1])
        self.actor_optimizer = tf.keras.optimizers.Adam(self.learning_rates[self.learning_rate_i][0])
        self.buffer = Buffer(self.n_states, self.n_actions, self.actor_optimizer, self.critic_optimizer, batch_size=self.batch_size)

        self.gamma = 0.99
        self.tau = 0.005
        self.ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(0.1) * np.ones(1))


    def __get_models(self, ep_balance = None):
        if ep_balance is not None:
            self.actor_main = AgentNetActor(self.n_states, weights='./checkpoints/actor_model_{}.h5'.format(ep_balance)).get_model()
            self.critic_main = AgentNetCritic(self.n_actions, self.n_states, weights='./checkpoints/critic_model_{}.h5'.format(ep_balance)).get_model()

            self.actor_target = AgentNetActor(self.n_states, weights='./checkpoints/target_actor_{}.h5'.format(ep_balance)).get_model()
            self.critic_target = AgentNetCritic(self.n_actions, self.n_states, weights='./checkpoints/target_critic_{}.h5'.format(ep_balance)).get_model()

        else:
            self.actor_main = AgentNetActor(self.n_states).get_model()
            self.critic_main = AgentNetCritic(self.n_actions, self.n_states).get_model()

            self.actor_target = AgentNetActor(self.n_states).get_model()
            self.critic_target = AgentNetCritic(self.n_actions, self.n_states).get_model()

            # Making the weights equal initially
            self.actor_target.set_weights(self.actor_main.get_weights())
            self.critic_target.set_weights(self.critic_main.get_weights())

    @tf.function
    def __update_target(self, target_weights, weights):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * self.tau + a * (1 - self.tau))

    def __policy(self, state, noise_object):
        sampled_actions = tf.squeeze(self.actor_main(state))
        noise = noise_object()
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise

        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, self.min_action, self.max_action)

        return [np.squeeze(legal_action)]


    def train(self):
        ep_balance = 5000

        for ep in range(1000):
            prev_state = self.env.reset()
            episodic_reward = 0            

            while True:
                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

                # Make a move
                action = self.__policy(tf_prev_state, self.ou_noise) #[tf.squeeze(self.actor_main(tf_prev_state)).numpy()] #

                # Recieve state and reward from environment.
                state, reward, done, info = self.env.step(action)
                self.env.render()

                # Store info in buffer
                self.buffer.record((prev_state, action, reward, state))
                episodic_reward += reward

                # Update model weights
                self.buffer.learn(self.actor_main, self.actor_target, self.critic_main, self.critic_target)
                self.__update_target(self.actor_target.variables, self.actor_main.variables)
                self.__update_target(self.critic_target.variables, self.critic_main.variables)

                # End this episode when done
                if done:
                    break

                prev_state = state

            if info != 'error':
                net = self.validate()
            else:
                net = 0

            # Save model if it's a best performer
            if net > 0:
                ep_balance = net
                tqdm.write('Saving weights, net worth: {}'.format(ep_balance))
                
                self.actor_main.save_weights('./checkpoints/actor_model_{:03.4g}.h5'.format(ep_balance))
                self.critic_main.save_weights('./checkpoints/critic_model_{:03.4g}.h5'.format(ep_balance))
                self.actor_target.save_weights('./checkpoints/target_actor_{:03.4g}.h5'.format(ep_balance))
                self.critic_target.save_weights('./checkpoints/target_critic_{:03.4g}.h5'.format(ep_balance))

                # if self.learning_rate_i >= len(self.learning_rates):
                #     break

                # actor_lr, critic_lr = self.learning_rates[self.learning_rate_i]
                # self.learning_rate_i += 1

                # # Reduce learning rate
                # critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
                # actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

                # self.buffer.actor_optimizer = actor_optimizer
                # self.buffer.critic_optimizer = critic_optimizer

    def validate(self, steps = None):
        init_balance = 1000
        balance_usd = init_balance
        balance_crypto = 1e-5
        if steps is None:
            steps = self.env.loader.steps

        loader = BinanceDataLoader(self.env.path, sample_size=self.sample_size, train_steps=steps)
        data_iterator = loader.process_data()
        self.env.pbar = tqdm(total=steps)
        nets = []
        prices = []

        data = next(data_iterator)
        init_price = data['price_old']
        while data:
            
            obs = data['observation']
            price = data['price_old']

            action = tf.squeeze(self.actor_main(np.array([obs]))).numpy()
            action = (action - 0.5) * 2

            # action = np.mean(np.gradient(obs)[-10:-1])

            # if action > 5:
            #     action = 1
            # elif action < -5:
            #     action = -1
            # else:
            #     action = action / 5

            # buy or hold
            if action >= 0:
                # reward *= (self.balance_0 / self.max_balance_0)
                transaction_ammount = action * balance_usd # * self.max_percentage_0 
                # Prevent trade if agent wants to spend less than one dollar
                if transaction_ammount > 10:
                    balance_usd -= transaction_ammount
                    balance_crypto += transaction_ammount / price

            # sell
            else:
                # reward *= (self.balance_1 / self.max_balance_1)
                transaction_ammount = action * balance_crypto # * self.max_percentage_1
                # Prevent trade if agent wants to spend less than one dollar
                if transaction_ammount * price < -10:
                    balance_crypto += transaction_ammount 
                    balance_usd -= transaction_ammount * price

            net = balance_usd + balance_crypto * price
            nets.append(net)
            prices.append(init_balance * price / init_price)

            self.env.pbar.set_postfix({'Net': net}, refresh=False)
            self.env.pbar.update(1)

            data = next(data_iterator)

        aoc_1 = np.trapz(np.array(nets) - 1000)
        aoc_2 = np.trapz(nets) - np.trapz(prices)

        aoc = int(max(aoc_1, aoc_2))

        tqdm.write('Environment reset!, Validation - Net worth: {}, Integral: {}'.format(net, aoc))

        plt.title("Integral: {}".format(aoc))
        plt.plot(prices, label='Price')
        plt.plot(nets, label='Net')
        plt.legend()
        plt.savefig('./plots/{}.png'.format(aoc))
        plt.clf()

        return aoc
            

            

