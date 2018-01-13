#coding:utf-8
import sys
sys.path.append("/usr/local/lib/python2.7/site-packages")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from PIL import Image

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray
def rgb2gray2(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

# 3D图标必须的模块，project='3d'的定义
from mpl_toolkits.mplot3d import Axes3D     

np.random.seed(42)

n_grids = 51        	# x-y平面的格点数 
c = n_grids / 2     	# 中心位置
nf = 2              	# 低频成分的个数

# 生成格点
x = np.linspace(0, 1, n_grids)
y = np.linspace(0, 1, n_grids)

# x和y是长度为n_grids的array
# meshgrid会把x和y组合成n_grids*n_grids的array，X和Y对应位置就是所有格点的坐标
X, Y = np.meshgrid(x, y)

# 生成一个0值的傅里叶谱
spectrum = np.zeros((n_grids, n_grids), dtype=np.complex)

# 生成一段噪音，长度是(2*nf+1)**2/2
noise = [np.complex(x, y) for x, y in np.random.uniform(-1,1,((2*nf+1)**2/2, 2))]

# 傅里叶频谱的每一项和其共轭关于中心对称
noisy_block = np.concatenate((noise, [0j], np.conjugate(noise[::-1])))

# 将生成的频谱作为低频成分
spectrum[c-nf:c+nf+1, c-nf:c+nf+1] = noisy_block.reshape((2*nf+1, 2*nf+1))

# 进行反傅里叶变换
Z = np.real(np.fft.ifft2(np.fft.ifftshift(spectrum)))

# 读取一张小白狗的照片并显示
plt.figure('A Little White Dog')
#little_dog_img = mpimg.imread('little_white_dog.png')
#little_dog_img = plt.imread('little_white_dog.PNG')
#plt.imshow(little_dog_img,cmap='gray')
#img2 = Image.open('little_white_dog.png').convert('LA')
#img2.save('greyscale.png')

img3 = mpimg.imread('little_white_dog.png')    
gray = rgb2gray(img3)    
plt.imshow(gray, cmap = plt.get_cmap('gray'))


# Z是上小节生成的随机图案，img0就是Z，img1是Z做了个简单的变换
img0 = Z
img1 = 3*Z + 4

# cmap指定为'gray'用来显示灰度图
fig = plt.figure('Auto Normalized Visualization')
ax0 = fig.add_subplot(121)
ax0.imshow(img0, cmap='gray')

ax1 = fig.add_subplot(122)
ax1.imshow(img1, cmap='gray')

plt.show()
