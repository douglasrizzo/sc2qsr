"""
SC2 Map Information -- :mod:`sc2qsr.sc2info.mapinfo`
****************************************************
"""

import os
import pickle
from pysc2 import maps, run_configs
from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import sc2api_pb2 as sc_pb
from . import data_dir

__filename = 'mapinfo'
__pickle = os.path.join(data_dir, __filename + '.pickle')

if os.path.isfile(__pickle):
    with open(__pickle, 'rb') as fp:
        __data = pickle.load(fp)
else:
    __data = {}
    with open(__pickle, 'wb') as fp:
        pickle.dump(__data, fp, protocol=pickle.HIGHEST_PROTOCOL)


def get_map_size(map_name: str) -> tuple:
    """Get the map size. If this info hasn't already been extracted by the agent before, a game will be started in order to get it. The information will then be pickled and further calls to this function will look for the info in the pickled file.

    :param map_name: the map name
    :type map_name: str
    :return: a tuple :math:`(x, y)` containing the dimensions of the map
    :rtype: tuple
    """
    if map_name in __data:
        map_size = __data[map_name]

    else:
        run_config = run_configs.get()
        map_inst = maps.get(map_name)

        with run_config.start(want_rgb=False) as controller:
            create = sc_pb.RequestCreateGame(
                local_map=sc_pb.
                LocalMap(map_path=map_inst.path, map_data=map_inst.data(run_config))
            )

            create.player_setup.add(type=sc_pb.Participant)
            join = sc_pb.RequestJoinGame(
                race=sc_common.Terran, options=sc_pb.InterfaceOptions(raw=True)
            )

            controller.create_game(create)
            controller.join_game(join)

            info = controller.game_info()
            map_size = info.start_raw.map_size

            __data[map_name] = (map_size.x, map_size.y)
            with open(__pickle, 'wb') as fp:
                pickle.dump(__data, fp, protocol=pickle.HIGHEST_PROTOCOL)

    return map_size
