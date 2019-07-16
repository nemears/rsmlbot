import numpy as np
import cv2
from mss.linux import MSS as mss
import mss.tools
import pyautogui as pag
from Xlib import display
import time
import rsmlbot as tools
import mousernn as mml
import lasagne as lasagne
import theano

seq_size = 5

network, train, val, predict = mml.build_mouse_model(seq_size)
with np.load('model.npz') as f:
     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
lasagne.layers.set_all_param_values(network['output'], param_values)
print('done loading model')

# finds closes fishing spot to player
def find_new_spot(image):
    fishim = cv2.imread('fish.png')
    fish_res = cv2.matchTemplate(image[:np.newaxis,:570],fishim,cv2.TM_CCOEFF_NORMED)
    fish_spots = np.where(fish_res >= 0.7)
    fish_loc = [pt for pt in zip(*fish_spots[::-1])]
    for f in fish_loc:
        f = list(f)
        f[0]+=13
        f[1]+=17
    return tuple(tools.closest_coord(fish_loc))

# moves mouse from start to end
def move_mouse(start, end, mon, click=True):
    done = False
    travel_list = []
    for i in range(seq_size+1):
        travel_list.append([start[0]/773,start[1]/534,end[0]/773,end[1]/534])
    temp_coord = predict([travel_list[-1 - seq_size:-1]])[0]
    print(temp_coord)
    temp_coord[0] = int(temp_coord[0]*773)
    temp_coord[1] = int(temp_coord[1]*534)
    while not done:
        pag.moveTo(mon['left']+temp_coord[0],mon['top']+temp_coord[1],0.016)
        travel_list.append([temp_coord[0]/773,temp_coord[1]/534,end[0]/773,
                            end[1]/534])
        if temp_coord is end:
            done = True
        x, junk = mml.sequential_data(travel_list,seq_size)
        temp_coord = predict(x)[-1]
        print(temp_coord,' ,',end[0]/773,' ',end[1]/534)
        temp_coord[0] = int(temp_coord[0]*773)
        temp_coord[1] = int(temp_coord[1]*534)
        #time.sleep(0.015)
    if click:
        pag.click()
        
# counts number of items in inventory
def inventory_num(image):
    item_count = 0
    for item in inv_items:
        res = cv2.matchTemplate(image[:np.newaxis,570:],item,cv2.TM_CCOEFF_NORMED)
        inv_spots = np.where(res >= 0.8)
        inv_loc = [pt for pt in zip(*inv_spots[::-1])]
        item_count += len(inv_loc)
    return item_count

# run bot to fish shrimp
def fish_shrimp():
    mon = tools.find_rs()
    if mon['width'] < 773 or mon['height'] < 534:
        print('rs not in screen or partially blocked!')
        return -1
    inv_items = [cv2.imread("net.png"), cv2.imread("shrimp.png"), cv2.imread("bottle.png")]
    not_fishing = True
    inventory = 0
    cursor_data = display.Display().screen().root.query_pointer()._data
    mouse_pos = (cursor_data["root_x"]-mon['left'],cursor_data["root_y"]-mon['top'])
    current_spot = (mon['left']+mon['width']/1.5,
                    mon['top'] + mon['height']/2)
    while True:
        imfile = 'imfile.png'
        shot = mss.mss().grab(mon)
        mss.tools.to_png(shot.rgb,shot.size,output=imfile)
        image = cv2.imread(imfile)
        if not_fishing and inventory < 28:
            #find closest fishing spot
            new_spot = find_new_spot(image)
            #move mouse and click
            move_mouse(current_spot, new_spot,mon)
            time.sleep(5)
            shot = mss.mss().grab(mon)
            mss.tools.to_png(shot.rgb,shot.size,output=imfile)
            image = cv2.imread(imfile)
            current_spot = find_new_spot(image)
            not_fishing = False
        else:
            #check if spot has changed
            temp_spot = find_new_spot(image)
            if temp_spot is not current_spot:
                not_fishing = True
            #count items
            inventory = inventory_num(image)
            time.sleep(2)
        if inventory > 28:
            #go to bank
            print('done fishing')

            
