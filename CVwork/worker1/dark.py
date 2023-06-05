import cv2
import numpy as np
def get_dark(I,k):
    I=np.array(I)
    I=np.min(I,axis=2)
    #直接最小值滤波
    kernel=np.ones((2*k-1,2*k-1))
    dark=cv2.erode(I,kernel)
    cv2.imshow('12233',dark)
    #cv2.imwrite('img_an',dark)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #print(dark)
    return dark

def larget_index(I,n):
    #排序
    flat=I.flatten()
    index=np.argpartition(flat,-n)[-n:]
    index=index[np.argsort(-flat[index])]
    return np.unravel_index(index,I.shape)

def get_A_dark(I,k=7):
    dark=get_dark(I,k)
    h,w=dark.shape
    num=(h*w)//1000
    #求A先排序
    index=larget_index(dark,num)
    A=np.ones(shape=(3,))
    for i in range(3):
        A[i]=np.max(I[index][i])

    return A,dark

def get_t(I,A,k=7,w=0.95):
    A=np.array(A)
    I=np.array(I)
    I=I / A
    I_dark=get_dark(I,k)
    t= 1- w*I_dark
    return t

# 导向滤波算法
def guideFilter(I, p, winSize, eps):
    I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
    I = I / np.max(I)
    mean_I = cv2.blur(I, winSize)
    mean_p = cv2.blur(p, winSize)
    mean_II = cv2.blur(I * I, winSize)
    mean_Ip = cv2.blur(I * p, winSize)
    var_I = mean_II - mean_I * mean_I
    cov_Ip = mean_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.blur(a, winSize)
    mean_b = cv2.blur(b, winSize)
    q = mean_a * I + mean_b
    return q

def get_image(I,A,t,t0=0.1):
    t=cv2.max(t,t0)
    J=np.ones_like(I)
    for n in range(I.shape[2]):
        J[:,:,n]=(I[:,:,n]-A[n])/t + A[n]
    return J

def RemHaze(I,t0=0.1,k=7,w=0.95):
    A,dark=get_A_dark(I,k)
    t=get_t(I,A,k,w)
    t=guideFilter(I,t,winSize=(20,20),eps=0.01)
    image=get_image(I,A,t,t0)
    return image

if __name__=="__main__":
    I=cv2.imread("img_2.png")
    #get_dark(I,20)
    image=RemHaze(I,0.5,7,0.2)
    cv2.imwrite('re4.png',image)






