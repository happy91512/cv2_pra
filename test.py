import cv2
import time

def count_time(func1):
    def wrap(*args,**kargs):
        start_time=time.time()
        func1()
        end_time=time.time()
        spend_time=end_time-start_time
        print("this function spent",spend_time,"second")
    
    return wrap

@count_time
def show_img(path,savePath):
    image1=cv2.imread(path,cv2.IMREAD_UNCHANGED)
    print(path,savePath)
    cv2.imshow("123",image1)
    cv2.imwrite(savePath,image1)
    cv2.waitKey(0)
    cv2.destroyWindow("123")
    return

read_path=str(input("Enter read path:"))
save_path=str(input("Enter save path:"))
show_img(read_path,save_path)

