from collections import defaultdict
import numpy as np

from factored_reps.envs.monte.ramops import getByte, bcd2int

class FakeMonteEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.timestep = 0
        self.state = dict({'timestep': -1})
        self.skull_dir = 'left'
        self.object_vertical_dir = 'up'
        self.lives = 5
        self.last_xy = np.asarray([-100, -100])
        self.should_terminate = False
        self.terminate_counter = 0

    def parseRAM(self, ram):
        """Get the current annotated RAM state dictonary”

        See RAM annotations:
        https://docs.google.com/spreadsheets/d/1KU4KcPqUuhSZJ1N2IyhPW59yxsc4mI4oSiHDWA3HCK4
        """
        state = dict()
        state['timestep'] = self.timestep
        state['frame'] = getByte(ram, '80')
        state['screen'] = getByte(ram, '83')
        state['level'] = getByte(ram, 'b9')
        state['screen_changing'] = getByte(ram, '84') != 0

        bcd_score = ''.join([format(getByte(ram, '9' + str(i)), '010b')[2:] for i in [3, 4, 5]])
        state['score'] = bcd2int(bcd_score)

        state['has_ladder'] = state['screen'] not in [5, 8, 12, 14, 15, 16, 17, 18, 20, 23]
        state['has_rope'] = state['screen'] in [1, 5, 8, 14]
        state['has_lasers'] = state['screen'] in [0, 7, 12]
        state['has_platforms'] = state['screen'] == 8
        state['has_bridge'] = state['screen'] in [10, 18, 20, 22]
        state['time_to_appear'] = getByte(ram, 'd3')
        state['time_to_disappear'] = -int(state['frame']) % 128 if state['time_to_appear'] == 0 else 0  # yapf: disable

        x = int(getByte(ram, 'aa'))
        y = int(getByte(ram, 'ab'))
        state['player_x'] = x
        state['player_y'] = y
        xy = np.asarray([x, y])
        # DEPRECATED! 'respawned' is only guaranteed to work for first screen; use 'respawning'
        state['respawned'] = 1 if (np.linalg.norm(self.last_xy - xy) > 10) else 0 # DEPRECATED!
        self.last_xy = xy

        state['player_jumping'] = 1 if getByte(ram, 'd6') != 0xFF else 0
        state['player_falling'] = 1 if getByte(ram, 'd8') != 0 else 0
        status = getByte(ram, '9e')
        status_codes = {
            0x00: 'standing',
            0x2A: 'running',
            0x3E: 'on-ladder',
            0x52: 'climbing-ladder',
            0x7B: 'on-rope',
            0x90: 'climbing-rope',
            0xA5: 'mid-air',
            0xBA: 'dead', # dive 1
            0xC9: 'dead', # dive 2
            0xC8: 'dead', # dissolve 1
            0xDD: 'dead', # dissolve 2
            0xFD: 'dead', # smoke 1
            0xE7: 'dead', # smoke 2
        }
        state['player_status'] = status_codes[status]

        look = int(format(getByte(ram, 'b4'), '08b')[-4])
        state['player_look'] = 'left' if look == 1 else 'right'

        state['lives'] = getByte(ram, 'ba')
        if (state['lives'] < self.lives):
            state['just_died'] = 1
        else:
            state['just_died'] = 0
        self.lives = state['lives']

        state['time_to_spawn'] = getByte(ram, 'b7')
        state['respawning'] = (state['time_to_spawn'] > 0 or state['player_status'] == 'dead')

        inventory = format(getByte(ram, 'c1'), '08b') # convert to binary
        possible_items = ['torch', 'sword', 'sword', 'key', 'key', 'key', 'key', 'hammer']
        state['inventory'] = [
            item for item, bit in zip(possible_items, inventory) if int(bit) == 1
        ]

        # yapf: disable
        objects = format(getByte(ram, 'c2'), '08b')[-4:]  # convert to binary; keep last 4 bits
        state['door_left'] = 'locked' if int(objects[0]) == 1 and state['screen'] in [1, 5, 17] else 'unlocked'
        state['door_right'] = 'locked' if int(objects[1]) == 1 and state['screen'] in [1, 5, 17] else 'unlocked'
        state['has_skull'] = int(objects[2]) if state['screen'] in [1, 5, 18] else 0 # skull screens
        if state['screen'] in [1, 5, 17]: # door screens
            state['has_object'] = int(objects[3])
        else:
            state['has_object'] = sum([int(c) for c in objects])
        # yapf: enable

        object_type = getByte(ram, 'b1')
        state['object_type'] = {
            0: 'none',
            1: 'jewel',
            2: 'sword',
            3: 'mallet',
            4: 'key',
            5: 'jump_skull',
            6: 'torch',
            8: 'snake',
            10: 'spider'
        }[object_type]
        object_configuration = int(format(getByte(ram, 'd4'), '08b')[-3:], 2)  # convert to binary; keep last 3 bits -- yapf: disable
        state['object_configuration'] = {
            0: 'one_single', # normal object
            1: 'two_near', # two objects, as close as possible
            2: 'two_mid', # same positions as three_near with center obj removed
            3: 'three_near', # same distance apart as two_near
            4: 'two_far', # same positions as three_mid with center obj removed
            5: 'one_double', # double-wide object
            6: 'three_mid', # same distance apart as two_mid
            7: 'one_triple', # triple-wide object
        }[object_configuration]
        state['has_spider'] = (state['has_object'] and state['object_type'] == 'spider')
        state['has_snake'] = (state['has_object'] and state['object_type'] == 'snake')
        state['has_jump_skull'] = (state['has_object'] and state['object_type'] == 'jump_skull')
        state['has_enemy'] = state['has_spider'] or state['has_snake'] or state['has_jump_skull']
        state['has_jewel'] = (state['has_object'] and state['object_type'] == 'jewel')

        state['object_x'] = int(getByte(ram, 'ac'))
        state['object_y'] = int(getByte(ram, 'ad'))
        state['object_y_offset'] = int(getByte(ram, 'bf')) # ranges from 0 to f
        obj_direction_bit = int(format(getByte(ram, 'b0'), '08b')[-4], 2)
        state['object_dir'] = 'right' if obj_direction_bit == 1 else 'left'

        skull_offset = defaultdict(lambda: 33, {
            18: [22,23,12][state['level']],
        })[state['screen']]  # yapf: disable
        state['skull_x'] = int(getByte(ram, 'af')) + skull_offset
        # Note: up to some rounding, player dies when |player_x - skull_x| <= 6
        if 'skull_x' in self.state.keys():
            if state['skull_x'] - self.state['skull_x'] > 0:
                self.skull_dir = 'right'
            if state['skull_x'] - self.state['skull_x'] < 0:
                self.skull_dir = 'left'
        state['skull_dir'] = self.skull_dir

        if 'object_y' in self.state.keys():
            if state['object_y'] - self.state['object_y'] > 0:
                self.object_vertical_dir = 'up'
            elif state['object_y'] - self.state['object_y'] < 0:
                self.object_vertical_dir = 'down'
        state['object_vertical_dir'] = self.object_vertical_dir

        self.state = state
        # return copy.deepcopy(self.state)

        self.timestep += 1
        return state