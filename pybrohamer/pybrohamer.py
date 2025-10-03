#! python3


# import csv
from enum import Enum
import math
import os

import matplotlib.pyplot as plt
import numpy
from scipy.stats import norm

from brispy.singlefile import SingleFile, SingleFileHorse, SingleFilePastPerformance, SingleFileRace, SingleFileRow
from horsedb2.variants import get_average_variants_db


AVERAGE_VARIANT: int = 17
DEFAULT_MIN_ROUTE_DISTANCE = 8


class BrohamerFigure(Enum):
    FR1 = 1
    FR2 = 2
    FR3 = 3
    EP = 4
    SP = 5
    AP = 6
    FX = 7
    E = 8
    TOTAL_T1 = 9
    TOTAL_T2 = 10
    TOTAL_T3 = 11


def get_average_variant(track: str, distance: float, surface: str, all_weather_flag: str):
    '''
    D - dirt
    T - turf
    d - inner turf
    t - outer turf
    others - not needed
    '''
    average_variant = 17
    averages = get_average_variants_db(track)
    if averages is None:
        return average_variant
    match surface:
        case 'D':
            # D could be dirt or tapeta so we need to check the all-weather flag
            if all_weather_flag == 'A':
                if distance < DEFAULT_MIN_ROUTE_DISTANCE:
                    average_variant = averages.iloc[0]['TapetaSprint']
                else:
                    average_variant = averages.iloc[0]['TapetaRoute']
            else:
                if distance < DEFAULT_MIN_ROUTE_DISTANCE:
                    average_variant = averages.iloc[0]['DirtSprint']
                else:
                    average_variant = averages.iloc[0]['DirtRoute']
        case 'T':
            if track == 'PRX':
                # PRX got rid of their turf course
                return average_variant
            elif track == 'AQU':
                if distance < DEFAULT_MIN_ROUTE_DISTANCE:
                    return averages.iloc[0]['DirtSprint']
                else:
                    return averages.iloc[0]['DirtRoute']
            if distance < DEFAULT_MIN_ROUTE_DISTANCE:
                average_variant = averages.iloc[0]['TurfSprint']
            else:
                average_variant = averages.iloc[0]['TurfRoute']
        case 'd':
            if distance < DEFAULT_MIN_ROUTE_DISTANCE:
                average_variant = averages.iloc[0]['Outer TurfSprint']
            else:
                average_variant = averages.iloc[0]['Outer TurfRoute']
        case 't':
            if distance < DEFAULT_MIN_ROUTE_DISTANCE:
                try:
                    average_variant = averages.iloc[0]['Inner TurfSprint']
                except KeyError:
                    try:
                        average_variant = averages.iloc[0]['Outer TurfSprint']
                    except Exception as e:
                        print(f'uh oh: {e}')
            else:
                try:
                    average_variant = averages.iloc[0]['Inner TurfRoute']
                except KeyError:
                    try:
                        average_variant = averages.iloc[0]['Outer TurfRoute']
                    except Exception as e:
                        print(f'uh oh: {e}')
    if not numpy.isnan(average_variant):
        return int(average_variant)
    else:
        return 17


def get_time_of_beaten_length(distance: float, time: float) -> float:
    FEET_IN_BEATEN_LENGTH = 9
    return FEET_IN_BEATEN_LENGTH / distance / time


class BrohamerPastPerformance:
    def __init__(self, name: str, sfpp: SingleFilePastPerformance):
        self.name = name
        self.distance = round(abs(sfpp.distance) / 220.0, 2)
        self.surface = sfpp.surface
        self.all_weather = True if sfpp.all_weather_surface_flag == 'A' else False
        self.track_condition = sfpp.track_condition
        self.track_code = sfpp.track_code
        self.finish_position = sfpp.finish_position
        self.t1 = sfpp.two_furlong_fraction if self.distance < DEFAULT_MIN_ROUTE_DISTANCE \
            else sfpp.four_furlong_fraction
        self.t2 = sfpp.four_furlong_fraction if self.distance < DEFAULT_MIN_ROUTE_DISTANCE \
            else sfpp.six_furlong_fraction
        self.t3 = sfpp.final_time
        self.track_variant = sfpp.track_variant
        self.average_variant = get_average_variant(self.track_code, self.distance, self.surface,
                                                   sfpp.all_weather_surface_flag)
        self.bl1 = sfpp.first_call_beaten_lengths
        self.bl2 = sfpp.second_call_beaten_lengths
        self.bl3 = sfpp.finish_beaten_lengths
        self.winner = 1 if self.finish_position == 1 else 0

        if abs(self.average_variant - self.track_variant) > 1:  # type: ignore
            if self.average_variant - self.track_variant > 4:
                adj3 = self.average_variant - 3 - self.track_variant
            else:
                if self.track_variant > self.average_variant:
                    diff = self.average_variant + 1 - self.track_variant
                    adj3 = math.floor(diff / 2)
                else:
                    adj3 = math.ceil((self.average_variant - 1 - self.track_variant) / 2)
        else:
            adj3 = 0
        if adj3 > 0:
            adj2 = math.floor(adj3 / 2)
        else:
            adj2 = math.ceil(adj3 / 2)
        if self.distance < DEFAULT_MIN_ROUTE_DISTANCE:
            if abs(adj3) > 3:
                adj1 = round(adj2 / 2, 0)
            else:
                adj1 = 0
        else:
            adj1 = round(2 * adj2 / 3, 0)

        # Leader's times adjusted for DRF Track Variant
        adj_t1 = self.t1 + 0.2 * adj1
        adj_t2 = self.t2 + 0.2 * adj2
        adj_t3 = self.t3 + 0.2 * adj3

        if self.distance < DEFAULT_MIN_ROUTE_DISTANCE:
            if not self.t1:
                self.fr1 = '-'
                self.fr2 = '-'
                self.fr3 = '-'
                self.ep = '-'
                self.sp = '-'
                self.ap = '-'
                self.fx = '-'
                self.energy = '-'
            else:
                self.fr1 = round((1320 - 10 * self.bl1) / adj_t1, 2)
                self.fr2 = round((1320 - 10 * (self.bl2 - self.bl1)) / (adj_t2 - adj_t1), 2)
                self.fr3 = round((660 * (self.distance - 4) - 10 * (self.bl3 - self.bl2)) / (adj_t3 - adj_t2), 2)
                self.ep = round((2640 - 10 * self.bl2) / adj_t2, 2)
                self.sp = round((self.ep + self.fr3) / 2, 2)
                self.ap = round((self.fr1 + self.fr2 + self.fr3) / 3, 2)
                self.fx = round((self.fr1 + self.fr3) / 2, 2)
                self.energy = round(self.ep / (self.ep + self.fr3), 4)
        else:
            if not self.t1:
                self.fr1 = '-'
                self.fr2 = '-'
                self.fr3 = '-'
                self.ep = '-'
                self.sp = '-'
                self.ap = '-'
                self.fx = '-'
                self.energy = '-'
            else:
                self.fr1 = round((2640 - 10 * self.bl1) / adj_t1, 2)
                self.fr2 = round((1320 - 10 * (self.bl2 - self.bl1)) / (adj_t2 - adj_t1), 2)
                self.fr3 = round((660 * (self.distance - 6) - 10 * (self.bl3 - self.bl2)) / (adj_t3 - adj_t2), 2)
                self.ep = round((3960 - 10 * self.bl2) / adj_t2, 2)
                self.sp = round((self.ep + self.fr3) / 2, 2)
                self.ap = round((self.fr1 + self.fr3) / 2, 2)
                self.fx = round((self.fr1 + self.fr3) / 2, 2)
                self.energy = round(self.ep / (self.ep + self.fr3), 4)

    def __str__(self):
        ret = ''
        for k, v in vars(self).items():
            ret += f'{k}={v}, '
        return f'BrohamerPastPerformance({ret[:-2]})'

    def __repr__(self):
        ret = ''
        for k, v in vars(self).items():
            ret += f'{k}={v}, '
        return f'BrohamerPastPerformance({ret[:-2]})'

    def is_winner(self) -> bool:
        return self.winner == 1


class BrohamerHorse:
    def __init__(self, sfh: SingleFileHorse):
        self.name = sfh.name
        self.post_position = sfh.post_position
        self.past_performances: list[BrohamerPastPerformance] = []
        if sfh.past_performances:
            for pp in sfh.past_performances:
                if not pp.track_code or int(pp.date[:4]) < 2025:
                    continue
                self.past_performances.append(BrohamerPastPerformance(self.name, pp))

    def __str__(self):
        ret = ''
        for k, v in vars(self).items():
            ret += f'{k}={v}, '
        return f'BrohamerHorse({ret[:-2]})'

    def __repr__(self):
        ret = ''
        for k, v in vars(self).items():
            ret += f'{k}={v}, '
        return f'BrohamerHorse({ret[:-2]})'


class BrohamerRace:
    def __init__(self, sfr: SingleFileRow):
        self.number = sfr.race.number
        self.horses: list[BrohamerHorse] = []

    def __str__(self):
        ret = ''
        for k, v in vars(self).items():
            ret += f'{k}={v}, '
        return f'BrohamerRace({ret[:-2]})'

    def __repr__(self):
        ret = ''
        for k, v in vars(self).items():
            ret += f'{k}={v}, '
        return f'BrohamerRace({ret[:-2]})'


def separate_races(sf: SingleFile, skip_maidens=False) -> list[BrohamerRace | None]:
    ret: list[BrohamerRace | None] = []
    current_race_number = 0
    for row in sf.rows:
        if current_race_number != row.race.number:
            current_race_number = row.race.number
            if is_maiden(row.race) and skip_maidens:
                ret.append(None)
                continue
            ret.append(BrohamerRace(row))
            assert len(ret) is current_race_number
        if is_maiden(row.race) and skip_maidens:
            continue
        assert ret[current_race_number - 1] is not None
        assert ret[current_race_number - 1].horses is not None  # type: ignore
        ret[current_race_number - 1].horses.append(BrohamerHorse(row.horse))  # type: ignore
    return ret


def get_average_brohamer_value(pps: list[BrohamerPastPerformance], figure: BrohamerFigure):
    match figure:
        case BrohamerFigure.FR1:
            return get_average_fr1(pps)
        case BrohamerFigure.FR2:
            return get_average_fr2(pps)
        case BrohamerFigure.FR3:
            return get_average_fr3(pps)
        case BrohamerFigure.EP:
            return get_average_ep(pps)
        case BrohamerFigure.SP:
            return get_average_sp(pps)
        case BrohamerFigure.AP:
            return get_average_ap(pps)
        case BrohamerFigure.FX:
            return get_average_fx(pps)
        case BrohamerFigure.E:
            return get_average_energy(pps)


def get_average_fr1(pps: list[BrohamerPastPerformance]) -> float:
    count = 0
    sum = 0
    for line in pps:
        if type(line.fr1) is float:
            count += 1
            sum += line.fr1
    return sum / count


def get_average_fr2(pps: list[BrohamerPastPerformance]) -> float:
    count = 0
    sum = 0
    for line in pps:
        if type(line.fr2) is not str:
            count += 1
            sum += float(line.fr2)
    return sum / count


def get_average_fr3(pps: list[BrohamerPastPerformance]) -> float:
    count = 0
    sum = 0
    for line in pps:
        if type(line.fr3) is not str:
            count += 1
            sum += float(line.fr3)
    return sum / count


def get_average_ep(pps: list[BrohamerPastPerformance]) -> float:
    count = 0
    sum = 0
    for line in pps:
        if type(line.ep) is not str:
            count += 1
            sum += float(line.ep)
    return sum / count


def get_average_sp(pps: list[BrohamerPastPerformance]) -> float:
    count = 0
    sum = 0
    for line in pps:
        if type(line.sp) is not str:
            count += 1
            sum += float(line.sp)
    return sum / count


def get_average_ap(pps: list[BrohamerPastPerformance]) -> float:
    count = 0
    sum = 0
    for line in pps:
        if type(line.ap) is not str:
            count += 1
            sum += float(line.ap)
    return sum / count


def get_average_fx(pps: list[BrohamerPastPerformance]) -> float:
    count = 0
    sum = 0
    for line in pps:
        if type(line.fx) is not str:
            count += 1
            sum += float(line.fx)
    return sum / count


def get_average_energy(pps: list[BrohamerPastPerformance]) -> float:
    count = 0
    sum = 0
    for line in pps:
        if type(line.energy) is not str:
            count += 1
            sum += float(line.energy)
    return sum / count


def get_stddev_brohamer_value(pps: list[BrohamerPastPerformance], figure: BrohamerFigure):
    match figure:
        case BrohamerFigure.FR1:
            return get_stddev_fr1(pps)
        case BrohamerFigure.FR2:
            return get_stddev_fr2(pps)
        case BrohamerFigure.FR3:
            return get_stddev_fr3(pps)
        case BrohamerFigure.EP:
            return get_stddev_ep(pps)
        case BrohamerFigure.SP:
            return get_stddev_sp(pps)
        case BrohamerFigure.AP:
            return get_stddev_ap(pps)
        case BrohamerFigure.FX:
            return get_stddev_fx(pps)
        case BrohamerFigure.E:
            return get_stddev_energy(pps)


def get_stddev_fr1(pps: list[BrohamerPastPerformance]) -> float:
    frs = [line.fr1 for line in pps if type(line.fr1) is float]
    return float(numpy.std(frs, ddof=1))


def get_stddev_fr2(pps: list[BrohamerPastPerformance]) -> float:
    frs = [line.fr2 for line in pps if type(line.fr2) is float]
    return float(numpy.std(frs, ddof=1))


def get_stddev_fr3(pps: list[BrohamerPastPerformance]) -> float:
    frs = [line.fr3 for line in pps if type(line.fr3) is float]
    return float(numpy.std(frs, ddof=1))


def get_stddev_ep(pps: list[BrohamerPastPerformance]) -> float:
    eps = [line.ep for line in pps if type(line.ep) is float]
    return float(numpy.std(eps, ddof=1))


def get_stddev_sp(pps: list[BrohamerPastPerformance]) -> float:
    sps = [line.sp for line in pps if type(line.sp) is float]
    return float(numpy.std(sps, ddof=1))


def get_stddev_ap(pps: list[BrohamerPastPerformance]) -> float:
    aps = [line.ap for line in pps if type(line.ap) is float]
    return float(numpy.std(aps, ddof=1))


def get_stddev_fx(pps: list[BrohamerPastPerformance]) -> float:
    fxs = [line.fx for line in pps if type(line.fx) is float]
    return float(numpy.std(fxs, ddof=1))


def get_stddev_energy(pps: list[BrohamerPastPerformance]) -> float:
    energies = [line.energy for line in pps if type(line.energy) is float]
    return float(numpy.std(energies, ddof=1))


def get_min_brohamer_value(pps: list[BrohamerPastPerformance], figure: BrohamerFigure) -> float:
    match figure:
        case BrohamerFigure.FR1:
            return get_min_fr1(pps)
        case BrohamerFigure.FR2:
            return get_min_fr3(pps)
        case BrohamerFigure.FR3:
            return get_min_fr3(pps)
        case BrohamerFigure.EP:
            return get_min_ep(pps)
        case BrohamerFigure.SP:
            return get_min_sp(pps)
        case BrohamerFigure.AP:
            return get_min_ap(pps)
        case BrohamerFigure.FX:
            return get_min_fx(pps)
        case BrohamerFigure.E:
            return get_min_energy(pps)
    raise ValueError('must use an acceptable BrohamerFigure')


def get_min_fr1(pps: list[BrohamerPastPerformance]) -> float:
    min = 10000.00
    for line in pps:
        if type(line.fr1) is float and line.fr1 < min:
            min = line.fr1
    return min


def get_min_fr2(pps: list[BrohamerPastPerformance]) -> float:
    min = 10000.00
    for line in pps:
        if type(line.fr2) is float and line.fr2 < min:
            min = line.fr2
    return min


def get_min_fr3(pps: list[BrohamerPastPerformance]) -> float:
    min = 10000.00
    for line in pps:
        if type(line.fr3) is float and line.fr3 < min:
            min = line.fr3
    return min


def get_min_ep(pps: list[BrohamerPastPerformance]) -> float:
    min = 10000.00
    for line in pps:
        if type(line.ep) is float and line.ep < min:
            min = line.ep
    return min


def get_min_sp(pps: list[BrohamerPastPerformance]) -> float:
    min = 10000.00
    for line in pps:
        if type(line.sp) is float and line.sp < min:
            min = line.sp
    return min


def get_min_ap(pps: list[BrohamerPastPerformance]) -> float:
    min = 10000.00
    for line in pps:
        if type(line.ap) is float and line.ap < min:
            min = line.ap
    return min


def get_min_fx(pps: list[BrohamerPastPerformance]) -> float:
    min = 10000.00
    for line in pps:
        if type(line.fx) is float and line.fx < min:
            min = line.fx
    return min


def get_min_energy(pps: list[BrohamerPastPerformance]) -> float:
    min = 10000.00
    for line in pps:
        if type(line.energy) is float and line.energy < min:
            min = line.energy
    return min


def get_max_brohamer_value(pps: list[BrohamerPastPerformance], figure: BrohamerFigure) -> float:
    match figure:
        case BrohamerFigure.FR1:
            return get_max_fr1(pps)
        case BrohamerFigure.FR2:
            return get_max_fr3(pps)
        case BrohamerFigure.FR3:
            return get_max_fr3(pps)
        case BrohamerFigure.EP:
            return get_max_ep(pps)
        case BrohamerFigure.SP:
            return get_max_sp(pps)
        case BrohamerFigure.AP:
            return get_max_ap(pps)
        case BrohamerFigure.FX:
            return get_max_fx(pps)
        case BrohamerFigure.E:
            return get_max_energy(pps)
    raise ValueError('must use an acceptable BrohamerFigure')


def get_max_fr1(pps: list[BrohamerPastPerformance]) -> float:
    max = 0.00
    for line in pps:
        if type(line.fr1) is float and line.fr1 > max:
            max = line.fr1
    return max


def get_max_fr2(pps: list[BrohamerPastPerformance]) -> float:
    max = 0.00
    for line in pps:
        if type(line.fr2) is float and line.fr2 > max:
            max = line.fr2
    return max


def get_max_fr3(pps: list[BrohamerPastPerformance]) -> float:
    max = 0.00
    for line in pps:
        if type(line.fr3) is float and line.fr3 > max:
            max = line.fr3
    return max


def get_max_ep(pps: list[BrohamerPastPerformance]) -> float:
    max = 0.00
    for line in pps:
        if type(line.ep) is float and line.ep > max:
            max = line.ep
    return max


def get_max_sp(pps: list[BrohamerPastPerformance]) -> float:
    max = 0.00
    for line in pps:
        if type(line.sp) is float and line.sp > max:
            max = line.sp
    return max


def get_max_ap(pps: list[BrohamerPastPerformance]) -> float:
    max = 0.00
    for line in pps:
        if type(line.ap) is float and line.ap > max:
            max = line.ap
    return max


def get_max_fx(pps: list[BrohamerPastPerformance]) -> float:
    max = 0.00
    for line in pps:
        if type(line.fx) is float and line.fx > max:
            max = line.fx
    return max


def get_max_energy(pps: list[BrohamerPastPerformance]) -> float:
    max = 0.00
    for line in pps:
        if type(line.energy) is float and line.energy > max:
            max = line.energy
    return max


def get_list_of_brohamer_value(pps: list[BrohamerPastPerformance], figure: BrohamerFigure):
    match figure:
        case BrohamerFigure.FR1:
            return get_list_of_fr1(pps)
        case BrohamerFigure.FR2:
            return get_list_of_fr2(pps)
        case BrohamerFigure.FR3:
            return get_list_of_fr3(pps)
        case BrohamerFigure.EP:
            return get_list_of_eps(pps)
        case BrohamerFigure.SP:
            return get_list_of_sps(pps)
        case BrohamerFigure.AP:
            return get_list_of_aps(pps)
        case BrohamerFigure.FX:
            return get_list_of_fxs(pps)
        case BrohamerFigure.E:
            return get_list_of_energy(pps)


def get_list_of_fr1(pps: list[BrohamerPastPerformance]) -> list[float]:
    return [line.fr1 for line in pps if type(line.fr1) is float]


def get_list_of_fr2(pps: list[BrohamerPastPerformance]) -> list[float]:
    return [line.fr2 for line in pps if type(line.fr2) is float]


def get_list_of_fr3(pps: list[BrohamerPastPerformance]) -> list[float]:
    return [line.fr3 for line in pps if type(line.fr3) is float]


def get_list_of_eps(pps: list[BrohamerPastPerformance]) -> list[float]:
    return [line.ep for line in pps if type(line.ep) is float]


def get_list_of_sps(pps: list[BrohamerPastPerformance]) -> list[float]:
    return [line.sp for line in pps if type(line.sp) is float]


def get_list_of_aps(pps: list[BrohamerPastPerformance]) -> list[float]:
    return [line.ap for line in pps if type(line.ap) is float]


def get_list_of_fxs(pps: list[BrohamerPastPerformance]) -> list[float]:
    return [line.fx for line in pps if type(line.fx) is float]


def get_list_of_energy(pps: list[BrohamerPastPerformance]) -> list[float]:
    return [line.energy for line in pps if type(line.energy) is float]


def is_maiden(race: SingleFileRace) -> bool:
    if 'Md' in race.classification:
        return True
    return False


def is_surface(race: SingleFileRace, target_surface: str, all_weather_flag: str = '') -> bool:
    if (race.surface == target_surface) and (race.todays_all_weather_surface_flag == all_weather_flag):
        return True
    return False


def is_different_type(d1: float, d2: float) -> bool:
    if (d1 < DEFAULT_MIN_ROUTE_DISTANCE) and (d2 < DEFAULT_MIN_ROUTE_DISTANCE):
        return False
    elif (d1 >= DEFAULT_MIN_ROUTE_DISTANCE) and (d2 >= DEFAULT_MIN_ROUTE_DISTANCE):
        return False
    return True


def plot_pdfs(race_number: str, starters_dict: dict, figure: BrohamerFigure):
    min = 1000.00
    max = 0
    for BrohamerRaces in starters_dict.values():
        tmp_min = get_min_brohamer_value(BrohamerRaces, figure)
        tmp_max = get_max_brohamer_value(BrohamerRaces, figure)
        if tmp_min < min:
            min = tmp_min
        if tmp_max > max:
            max = tmp_max
    x = numpy.linspace(min - 3, max + 3)
    for starter, BrohamerRaces in starters_dict.items():
        if len(BrohamerRaces) < 4:
            continue
        mean = get_average_brohamer_value(BrohamerRaces[:4], figure)
        stddev = get_stddev_brohamer_value(BrohamerRaces[:4], figure)
        pdf = norm.pdf(x, loc=mean, scale=stddev)
        plt.plot(x, pdf, label=f'{starter}')
    plt.title(f'PDF of race {race_number}')
    plt.xlabel('')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.show()


def to_csv(pp: BrohamerPastPerformance, output: str, mode: str = 'a', newline: str = ''):
    if mode not in ('a', 'w'):
        return None
    if mode == 'w' and os.path.exists(output):
        return None
    if mode == 'a' and not os.path.exists(output):
        return None
    # try:
    #     with open(output, mode=mode, newline=newline) as f:
    #         pass
    # except FileNotFoundError:
    #     print('the output path was not found')


def create_csv(output: str, newline: str = ''):
    if os.path.exists(output):
        return None
    # with open(output, mode='w+', newline=newline) as f:
    #     headers = []


__all__ = [
    'BrohamerRace',
    'separate_races',
    'to_csv',
    'plot_pdfs',
]
