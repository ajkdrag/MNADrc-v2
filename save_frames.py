import cv2
import os
from multiprocessing import Pool
import sys

directory = sys.argv[1]

def mk_frames(vid):
        print(vid)
        os.mkdir("/".join(vid.split("/")[:-1])+'/frames/'+vid[:-4])
        cap = cv2.VideoCapture(vid)
        i = 0
        ret = True
        while(cap.isOpened()):
                ret, frame = cap.read()
                if ret == False:
                        break
                cv2.imwrite('frames'+'/'+vid[:-4]+'/'+str(i)+'.jpg',frame)
                i+=1
	 
        cap.release()
        cv2.destroyAllWindows()


p = Pool(int(sys.argv[2]))
file_list = os.listdir(directory)
p.map(mk_frames,file_list)
