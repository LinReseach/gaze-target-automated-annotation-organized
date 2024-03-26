import pandas as pd

datafolder = './data_fin/'
datafile = 'pixel_position_vis.txt'
path = datafolder+datafile

def readbbtxt(path=path):
    # tl:topleft coordinate, br: bottom right coordinate
    data = pd.read_csv(path,
                    sep=', [^0-9]',
                    names=['file',
                            'tl_robot_x',
                            'br_robot_x',
                            'tl_block_x',
                            'br_block_x',
                            'obj_list'],

                    engine='python')

    ppy = ['tl_robot_y',
        'br_robot_y',
        'tl_block_y',
        'br_block_y']
    
    ppx = ['tl_robot_x',
        'br_robot_x',
        'tl_block_x',
        'br_block_x']
   
    for py, px in zip(ppy, ppx):
        data[py] = data[px].apply(lambda x : float(x.split(', ')[1].replace(')', ''))).astype(float)
        data[px] = data[px].apply(lambda x : float(x.split(', ')[0])).astype(float)
        data['obj_list'] = data['obj_list'].apply(lambda x: str(x))
        data['obj_list'] = data['obj_list'].apply(lambda x : '[' + x if x[-1] == ']' and x[0] != '[' else '[]' if x == 'one' else x)
    
    # for i in data.keys():
    #     print(data[i])
    #     data[i] = data[i].tolist()

    
    return data
