import pandas as pd

datafolder = './data_fin/'
datafile = 'pixel_position_invis_new.txt'
path = datafolder + datafile


def readbbtxt(path=path):
    # tl:topleft coordinate, br: bottom right coordinate
    data = pd.read_csv(path,
                       sep=', [^0-9]',
                       names=['file',
                              'tl_tablet_x',
                              'br_tablet_x',
                              'tl_robot_x',
                              'br_robot_x',
                              'tl_pp_x',
                              'br_pp_x'],
                       usecols=range(0, 7),
                       engine='python')

    ppy = ['tl_tablet_y',
           'br_tablet_y',
           'tl_robot_y',
           'br_robot_y',
           'tl_pp_y',
           'br_pp_y']

    ppx = ['tl_tablet_x',
           'br_tablet_x',
           'tl_robot_x',
           'br_robot_x',
           'tl_pp_x',
           'br_pp_x']

    for py, px in zip(ppy, ppx):
        data[py] = data[px].apply(lambda x: float(x.split(', ')[1].replace(')', '')))
        data[px] = data[px].apply(lambda x: float(x.split(', ')[0]))
        # data['obj_list'] = data['obj_list'].apply(lambda x: str(x))
        # data['obj_list'] = data['obj_list'].apply(
        #     lambda x: '[' + x if x[-1] == ']' and x[0] != '[' else '[]' if x == 'one' else x)

    # for i in data.keys():
    #     print(i, data[i])

    return data
