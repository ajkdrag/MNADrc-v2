import cv2
import os

directory = r'C:\Users\kevin\OneDrive\Desktop\training\videos'
for filename in os.listdir(directory):
        print(filename)
        os.mkdir(directory+'/frames/'+filename)
        cap = cv2.VideoCapture(filename)
        i = 0
        ret = True
        while(cap.isOpened()):
                ret, frame = cap.read()
                if ret == False:
                        break
                cv2.imwrite('frames'+'/'+filename+'/'+str(i)+'.jpg',frame)
                i+=1
	 
        cap.release()
        cv2.destroyAllWindows()
