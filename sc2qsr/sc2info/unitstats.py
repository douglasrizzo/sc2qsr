"""
Unit Statistics from Liquipedia -- :mod:`sc2qsr.sc2info.unitstats`
******************************************************************
"""
import os
import pickle
from . import data_dir

__filename = 'unitstats'
__json = os.path.join(data_dir, __filename + '.json')
__pickle = os.path.join(data_dir, __filename + '.pickle')
__csv = os.path.join(data_dir, __filename + '.csv')


def radius(unit_type: int) -> float:
    """Return the radius of a unit, given its unit type ID

    :param unit_type: the unit type ID, according to :mod:`pysc2.lib.stats`
    :type unit_type: int
    :return: the unit's radius
    :rtype: float
    """
    return __data['Unit Radius'][unit_type]


def attack_range(unit_type: int):
    """Return the attack range of a unit, given its unit type ID

    :param unit_type: the unit type ID, according to :mod:`pysc2.lib.stats`
    :type unit_type: int
    :return: the unit's attack range
    :rtype: float
    """
    return __data['Range Attack 1'][unit_type]


def sight(unit_type: int):
    """Return the sight range of a unit, given its unit type ID

    :param unit_type: the unit type ID, according to :mod:`pysc2.lib.stats`
    :type unit_type: int
    :return: the unit's sight range
    :rtype: float
    """
    return __data['Sight'][unit_type]


if __name__ == "__main__" or not os.path.isfile(__pickle):
    import pandas as pd
    with open(__pickle, 'wb') as fp:
        pickle.dump(
            pd.read_csv(
                __csv,
                header=0,
                index_col='Unit Type ID',
                usecols=['Unit Type ID', 'Unit Radius', 'Range Attack 1', 'Sight']
            ).to_dict(),
            fp,
            protocol=pickle.HIGHEST_PROTOCOL
        )

with open(__pickle, 'rb') as fp:
    __data = pickle.load(fp)
