"""
Agents -- :mod:`sc2qsr.agents`
******************************
"""
import os
import re
from abc import abstractmethod

import networkx as nx
import numpy as np
from pysc2.env.environment import TimeStep
from pysc2.lib import actions, features

from .rl import QLearning
from .sc2info.unitstats import radius
from .spatial import qualitative as qsr


class BaseAgent:

    def __init__(self):
        self._base_state = None
        self._oo_state = None
        self._graph_state = None
        self._is_oo_state_synchronized = True
        self._is_graph_state_synchronized = True

    def set_new_base_state(self, state):
        self._base_state = state
        self._is_oo_state_synchronized = False
        self._is_graph_state_synchronized = False

    @abstractmethod
    def get_oo_state(self, base_state=None):
        pass

    @staticmethod
    def oo_to_graph(oo_state):
        return nx.complete_graph(oo_state)

    def get_graph_state(self):
        if not self._is_graph_state_synchronized:
            self._graph_state = BaseAgent.oo_to_graph(self.get_oo_state())

        return self._graph_state


class PySC2Agent(BaseAgent):
    """An agent that uses qualitative spatial reasoning to discretize states received by the pysc2 environment, select actions that are spatially qualitative and convert them back to pysc2 actions.

    :param map_size: tuple containing the (x,y) dimensions of the map
    :type map_size: tuple
    :param m: granularity for qualitative spatial angular regions, defaults to 4
    :type m: int, optional
    :param n: granularity for qualitative spatial distance regions, defaults to 4
    :type n: int, optional
    :param alpha: learning rate, defaults to .1
    :type alpha: float, optional
    :param gamma: discount factor, defaults to .6
    :type gamma: float, optional
    :param epsilon: value for epsilon-greedy policy, defaults to .1
    :type epsilon: float, optional
    :param policy: what type of policy to use from ``['egreedy', 'boltzmann']``, defaults to 'egreedy'
    :type policy: str, optional
    """

    _NO_ARGUMENT_ACTIONS = ['no_op']
    _POINT_ACTIONS = ['Attack_pt', 'Move_pt']
    _UNIT_ACTIONS = []

    def __init__(
        self,
        map_size: tuple,
        m: int = 4,
        n: int = 4,
        alpha=.1,
        gamma=.6,
        epsilon=.1,
        policy='egreedy'
    ):
        super(PySC2Agent, self).__init__()
        self.__mapsize = map_size
        self.__m = m
        self.__n = n

        self.__generate_qualitative_actions()

        # TODO check if the greek symbols actually represent that
        self.q_table = QLearning(
            actions=len(self.available_actions),
            policy=policy,
            learning_rate=alpha,
            reward_decay=gamma,
            epsilon=epsilon
        )

        self.__action = None
        self.__state = None
        self.__previous_state = None

    def __generate_qualitative_actions(self):
        # create a list of actions with no arguments
        self.available_actions = [
            (action_name, ()) for action_name in PySC2Agent._NO_ARGUMENT_ACTIONS
        ]

        # discretize point actions to refer to entire qualitative regions
        for pa in PySC2Agent._POINT_ACTIONS:
            for m in range(self.__m):
                for n in range(self.__n):
                    self.available_actions.append((pa, m, n))

        # TODO think about how to program unit actions
        # pysc2_unit_actions = []

    def __qualitative2pysc_action(self, action_name: str, unit_tag, unit_type, unit_x, unit_y):
        # get values for m,n
        qdir, qdist = map(int, action_name.split("_")[-2:])

        # pass necessary values to conversion function
        # transform an (m, n) sector into a x, y coordinates
        x, y = qsr.epra2cart(qdir, qdist, self.__m, self.__n, radius(unit_type))

        # remove _m_n from the end of the action name
        act_name = "_".join(action_name.split('_')[:-2])

        # build the action object and return it
        actions.RAW_FUNCTIONS[act_name]("now", unit_tag, (unit_x + x, unit_y + y))
        pass

    @staticmethod
    def array2oneliner(a: np.ndarray) -> str:
        """Converts a `numpy.ndarray` into a one-line string containing all its information

        :param a: A 1D or 2D `numpy.ndarray`
        :type a: numpy.ndarray
        :return: a one-line string representation of the array
        :rtype: str
        """
        text = np.array2string(a)
        reps = [
            ('\[\[ +|\]\]', ''),  # remove double brackes from start and end of 2d array
            (']\n +\[ +', '|'),  # replace single brackets and newline with pipes in 2d array
            ('\n', ' '),  # replace newline with spaces in 1d array
            ('\[|\]', ' '),  # remove remaining brackets from 1d array
            ("  +", " ")  # replace multiple spaces with a single space
        ]  # noqa

        # apply each replacement in sequence
        for rep in reps:
            pattern = re.compile(rep[0])
            text = pattern.sub(lambda m: rep[1], text)

        return text.strip()

    @staticmethod
    def oneliner2array(s: str, dtype):
        """Converts a one-line string representation (see :func:`array2oneliner`)
        of one or more numpy arrays back into their string representations

        :param s: one-line string representation of the arrays
        :type s: str
        :param dtype: a function which is called to convert the value of each element in the arrays
        :type dtype: function
        :return: a tuple of arrays if `s` contains more than one array. Otherwise, a single array.
        """
        # I didn't aim for readibility here
        # first, we split the string s into multiple substrings a, representing a single 1D or 2D array
        # then, we split each string a into multiple substrings l, representing a single row in the array
        # then, we split each string l into multiple elements e, representing a single element
        # then, we cast e to dtype
        # each a is transformed into a numpy array
        # e is a list all 1D or 2D arrays contained in s
        e = [
            np.array([[dtype(e) for e in l.split(' ')] for l in a.split('|')])
            for a in s.split('#')
        ]

        return tuple(e) if len(e) > 1 else e[0]

    @staticmethod
    def get_elevations_by_unit_type(units: list) -> list:
        # according to Moratz 2012, (radius / 2) is the distance
        # of the second qualitative distance sector to the elevated
        # point. however, in order for the first region to represent
        # the actual location an entity occupies, this distance
        # should be equal to the radius
        # TODO I need to figure that out
        return [radius(u) for u in units]

    @staticmethod
    def get_my_units_by_type(obs: TimeStep, unit_type):
        return [
            unit for unit in obs.observation.raw_units
            if unit.unit_type == unit_type and unit.alliance == features.PlayerRelative.SELF
        ]

    @staticmethod
    def get_enemy_units_by_type(obs: TimeStep, unit_type):
        return [
            unit for unit in obs.observation.raw_units
            if unit.unit_type == unit_type and unit.alliance == features.PlayerRelative.ENEMY
        ]

    @staticmethod
    def get_my_units(obs: TimeStep):
        return [
            unit for unit in obs.observation.raw_units
            if unit.alliance == features.PlayerRelative.SELF
        ]

    @staticmethod
    def get_enemy_units(obs: TimeStep):
        return [
            unit for unit in obs.observation.raw_units
            if unit.alliance == features.PlayerRelative.ENEMY
        ]

    @staticmethod
    def get_all_units(obs: TimeStep):
        return obs.observation.raw_units

    def state_from_obs(self, obs: TimeStep):
        my_collection = self.extract_features_from_obs(obs)

        unit_types = my_collection[:, 1]
        coordinates = my_collection[:, 2:4]
        elevations = self.get_elevations_by_unit_type(unit_types)

        quali_dirs, quali_dists = qsr.generate_qualitative_configuration(
            coordinates, self.__m, self.__n, elevations
        )

        return self.array2oneliner(np.concatenate((my_collection[:, 0:2], quali_dists), axis=1)
                                   ) + '#' + self.array2oneliner(quali_dirs)

    def relative_state_from_obs(self, obs: TimeStep):
        my_collection = self.extract_features_from_obs(obs)

        unit_types = my_collection[:, 1]
        coordinates = my_collection[:, 2:4]
        center = (self.__mapsize[0] / 2, self.__mapsize[1] / 2)

        # TODO what elevation should I use for the reference?
        # probably related to the map size
        # remember that the goal is to transfer, so the elevation should be something
        # meaningful in multiple domains
        elevation = 1

        rel_quali_config = qsr.qualitative_with_reference(
            coordinates, self.__m, self.__n, elevation, ref=center
        )

        return self.array2oneliner(
            np.concatenate((my_collection[:, 0:2], rel_quali_config), axis=1)
        )

    @staticmethod
    def extract_features_from_units(units, sort=True):
        my_collection = np.array(
            [[unit.alliance, unit.unit_type, unit.x, unit.y] for unit in units]
        )

        if sort:
            my_collection.sort(axis=0)

        return my_collection

    @staticmethod
    def extract_features_from_obs(obs: TimeStep, sort=True):
        units = PySC2Agent.get_all_units(obs)
        return PySC2Agent.extract_features_from_units(units, sort)

    def step(self, obs: TimeStep):
        # The step method is the core part of our agent, it’s where all of our decision making takes place. At the
        # end of every step you must return an action
        super().step(obs)

        if (self.episodes + 1) % 100 == 0:
            os.system('clear')
            print("episode: {}, reward: {}".format(self.episodes + 1, self.reward))

        # the step method is a little weird
        # first, we process the observed reward and update the Q table,
        # then we select an action and apply it

        # store the previous state and build the qualitative state representation,
        # by extracting the relevant features from the observation
        self.__previous_state = self.__state
        self.__state = self.state_from_obs(obs)

        # only update the Q table if we have already interacted with the environment
        if self.__action is not None:
            self.q_table.learn(self.__previous_state, self.__action, obs.reward, self.__state)

        # action selection according to policy
        self.__action = self.q_table.choose_action(self.__state)

        # return actions.RAW_FUNCTIONS.Build_Pylon_pt("now", probe.tag, pylon_xy)
        # return actions.RAW_FUNCTIONS.Build_Gateway_pt("now", probe.tag, gateway_xy)
        # return actions.RAW_FUNCTIONS.Train_Zealot_quick("now", gateway.tag)
        # return actions.RAW_FUNCTIONS.Attack_pt("now", zealot.tag, (attack_x, attack_y))
        # Take action
        # return []
        return self.__qualitative2pysc_action(self.available_actions[self.__action])


if __name__ == "__main__":
    from absl import app
    from pysc2.env import run_loop, sc2_env
    from pysc2.lib.actions import ActionSpace
    from pysc2.lib.features import AgentInterfaceFormat
    from sc2qsr.sc2info.mapinfo import get_map_size

    map_name = 'DefeatRoaches'  # map name
    players = [
        sc2_env.Agent(sc2_env.Race.terran),
        sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.very_easy)
    ]  # list of players in the environment, at least one agent is necessary

    agent_interface_format = AgentInterfaceFormat(action_space=ActionSpace.RAW, use_raw_units=True)
    step_mul = 8  # agent step will execute once every X step_mul, default is 8
    game_steps_per_episode = 0  # no limit for games
    save_replay_episodes = 1
    replay_dir = '/home/dodo/sc2qsr Replays'

    def main(unused_argv):
        agent = PySC2Agent(get_map_size(map_name))

        try:
            while True:
                # to get a list of map names
                # python -m pysc2.bin.map_list
                with sc2_env.SC2Env(
                    map_name=map_name,
                    players=players,
                    agent_interface_format=agent_interface_format,
                    step_mul=step_mul,
                    game_steps_per_episode=game_steps_per_episode,
                    save_replay_episodes=save_replay_episodes,
                    replay_dir=replay_dir
                ) as env:
                    run_loop.run_loop([agent], env)

        except KeyboardInterrupt:
            pass

    app.run(main)
