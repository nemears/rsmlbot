import numpy as np
import cv2
from mss.linux import MSS as mss
import mss.tools
import pyautogui as pag
import csv
import time
from pynput import mouse
from Xlib import display

def get_monitor():
    with mss.mss() as sct:
        size = pag.size()
        mon = {"top": 0 , "left": 0, "width": size.width, "height": size.height}
        return np.array(sct.grab(mon))

def find_top_corner():
    with mss.mss() as sct:
        cmon = cv2.imread(sct.shot(),0)
        rl = cv2.imread('rl_logo.png',0)
        res = cv2.matchTemplate(cmon, rl ,cv2.TM_CCOEFF)
        return cv2.minMaxLoc(res)[3]

def sim_pixel(a,b):
    val1 = a[0] < b[0]+5 and a[0] > b[0]-5
    val2 = a[1] < b[1]+5 and a[1] > b[0]-5
    val3 = a[2] < b[2]+5 and a[2] > b[2]-5
    return val1 and val2 and val3

def find_rs():
    no_rs = True
    valr = (0,0,0,0)
    while no_rs:
        left, top = find_top_corner()
        # not just finding it in sidebar
        if left >= 67:
            img = get_monitor()
            height = 0
            cpixel = img[top][left]
            while sim_pixel(cpixel, img[top][left]):
                height +=1
                cpixel = img[top+height][left]
            width = 0
            cpixel = img[top][left]
            while sim_pixel(cpixel, img[top][left]):
                width+=1
                cpixel = img[top][left+width]
            no_rs = False
            valr = {"left":left,"top":top,"width":width,"height":height}
    return valr

def closest_coord(coords):
    center = (272,392)
    mDist = 10000
    mcoord = (-1,-1)
    for c in coords:
        cDist = ((c[0] - center[0])**2 + (c[1]-center[1])**2)**0.5
        if cDist < mDist:
            mDist = cDist
            mcoord = c
    return mcoord

def video_to_table(path):
    video = cv2.VideoCapture(path)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    cursor = cv2.imread("cursor.png")
    fish = cv2.imread("fish.png")
    inv_items = [cv2.imread("net.png"), cv2.imread("shrimp.png"), cv2.imread("bottle.png")]
    click = cv2.imread("click.png")
    with open(path[:-4] + '.csv', 'w', newline = '') as file:
        fieldnames = ['cursor_loc', 'fish_locs', 'inv_num', 'click']
        writer = csv.DictWriter(file, fieldnames = fieldnames)
        success = True
        count = 0
        print('0 %')
        last_click = -20
        while success:
            success, image = video.read()
            if not success:
                break
            #cursor location convolution
            cursor_res = cv2.matchTemplate(image,cursor,cv2.TM_CCOEFF)
            cursor_loc = cv2.minMaxLoc(cursor_res)[3]
            #fishing spots convolution
            fish_res = cv2.matchTemplate(image[:np.newaxis,:570],fish,cv2.TM_CCOEFF_NORMED)
            fish_spots = np.where(fish_res >= 0.7)
            fish_loc = [pt for pt in zip(*fish_spots[::-1])]
            for f in fish_loc:
                f = list(f)
                f[0]+=13
                f[1]+=17
            close_fish = tuple(closest_coord(fish_loc))
            #inventory number
            item_count = 0
            for item in inv_items:
                res = cv2.matchTemplate(image[:np.newaxis,570:],item,cv2.TM_CCOEFF_NORMED)
                inv_spots = np.where(res >= 0.8)
                inv_loc = [pt for pt in zip(*inv_spots[::-1])]
                item_count += len(inv_loc)
            #click
            click_res = cv2.matchTemplate(image,click,cv2.TM_CCOEFF_NORMED)
            click_loc = (0,0)
            if np.max(click_res) > 0.7 and count >= last_click + 20:
                click_loc = 1
                last_click = count
            else:
                click_loc = 0
            #screen turn (hard, do later :P)
            
            #write to csv
            writer.writerow({'cursor_loc': cursor_loc, 'fish_locs': close_fish, 'inv_num': item_count, 'click': click_loc})
            count+=1
            if (count/num_frames)*10%1 == 0:
                print(100*count/num_frames, '%')

click = False # global var for thread

def on_click(x,y,button,pressed):
    global click
    if pressed:
        click = True

def live_play_to_table(name):
    global click
    click = False
    mon = find_rs()
    cursor = cv2.imread("cursor.png")
    fish = cv2.imread("fish.png")
    inv_items = [cv2.imread("net.png"), cv2.imread("shrimp.png"), cv2.imread("bottle.png")]
    with open(name + ".csv", 'w', newline = '') as file:
        fieldnames = ['cursor_locx','cursor_locy', 'fish_locx', 'fish_locy',
                      'inv_num', 'click','time_elapsed']
        writer = csv.DictWriter(file, fieldnames = fieldnames)
        count = 0
        prev_time = time.time()
        with mouse.Listener(on_click=on_click) as listener:
            while 1:
                #get image
                imfile = 'imfile.png'
                shot = mss.mss().grab(mon)
                mss.tools.to_png(shot.rgb,shot.size,output=imfile)
                image = cv2.imread(imfile)
                #cursor location
                cursor_data = display.Display().screen().root.query_pointer()._data
                cursor_loc = (cursor_data["root_x"]-mon['left'],cursor_data["root_y"]-mon['top'])
                #fishing spots convolution
                fish_res = cv2.matchTemplate(image[:np.newaxis,:570],fish,cv2.TM_CCOEFF_NORMED)
                fish_spots = np.where(fish_res >= 0.7)
                fish_loc = [pt for pt in zip(*fish_spots[::-1])]
                for f in fish_loc:
                    f = list(f)
                    f[0]+=13
                    f[1]+=17
                close_fish = tuple(closest_coord(fish_loc))
                #inventory number
                item_count = 0
                for item in inv_items:
                    res = cv2.matchTemplate(image[:np.newaxis,570:],item,cv2.TM_CCOEFF_NORMED)
                    inv_spots = np.where(res >= 0.8)
                    inv_loc = [pt for pt in zip(*inv_spots[::-1])]
                    item_count += len(inv_loc)
                #click
                click_loc = 0
                if click:
                    click_loc = 1
                #screen turn (hard, do later :P)
                time_elapsed = time.time()-prev_time
                prev_time = time.time()
                #write to csv
                writer.writerow({'cursor_locx': cursor_loc[0]/mon['width'],
                                 'cursor_locy': cursor_loc[1]/mon['height'],
                                 'fish_locx': close_fish[0]/mon['width'],
                                 'fish_locy': close_fish[1]/mon['height'],
                                 'inv_num': item_count/28, 'click': click_loc,
                                 'time_elapsed': time_elapsed})
                count+=1
                click = False
            listener.join()
            
def mouse_to_table(path,mon):
    print('1')
    time.sleep(1)
    print('2')
    time.sleep(1)
    print('3')
    time.sleep(1)
    print('Go!')
    data = []
    i = 0
    prev_time = time.time()
    while i < 100:
        data.append
        cursor_data = display.Display().screen().root.query_pointer()._data
        data.append([cursor_data["root_x"]-mon['left'],
                     cursor_data["root_y"]-mon['top'],
                     time.time()-prev_time])
        i+=1
        prev_time = time.time()
    with open(path+'.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for d in data:
           writer.writerow(d) 
