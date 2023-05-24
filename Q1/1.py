import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import os

##for(a)
def subplot(points , result1 , result2 , img):


    plt.imshow(img)
    plt.scatter(points[:, 0], points[:, 1],  s=0.5)
    plt.plot(result1[:, 0], result1[:, 1], 'b-' ,linewidth=0.5)

    plt.plot(result2[:, 0], result2[:, 1], 'r-' ,linewidth=0.5)
    plt.savefig('./output/1a.png')
    plt.close()

##for(b)
def plot(points , result , img):
    plt.imshow(img)
    plt.scatter(points[:, 0], points[:, 1],  s=5)
    plt.plot(result[:, 0], result[:, 1], 'r-' ,linewidth=0.5)
    plt.savefig('./output/1b.png')
    plt.close()

def bezier_curve(points,setting):
    basis_matrix = np.array([[-1, 3, -3, 1],
                             [3, -6, 3, 0],
                             [-3, 3, 0, 0],
                             [1, 0, 0, 0]])
    result = np.empty((0, 2), float)
    if setting=='low_detail':
        step = 0.5
    else:
        step = 0.01
    for i in range(0, len(points)-3, 3):
        p0 = points[i]
        p1 = points[i+1]
        p2 = points[i+2]
        p3 = points[i+3]
        int_step = int(step*100)
        for t in range(0, 100+int_step, int_step):
            my_t = t/100
            t_matrix = np.array([[my_t**3, my_t**2, my_t, 1]])
            p = np.dot(np.dot(t_matrix, basis_matrix), np.array([p0, p1, p2, p3]))
            result = np.append(result, p, axis=0)
    print(result.shape)
    return result

def bilinear_interpolation(img,n):
    height, width, channel = img.shape
    new_height = height * n
    new_width = width * n
    res = np.zeros((new_height, new_width, channel), dtype=np.uint8)
    
    def get_coef(lu,ld,ru,rd,i,j,n):
        orig_i = i/n
        orig_j = j/n
        h1 = orig_i - lu[0]
        h2 = ld[0] - orig_i
        w1 = orig_j - lu[1]
        w2 = ru[1] - orig_j
        return h1*w1, h1*w2, h2*w1, h2*w2
    
    for i in tqdm(range(new_height)):
        for j in range(new_width):
            lu = np.floor([i/n,j/n]).astype(int)
            ru = np.floor([i/n,j/n+1]).astype(int)
            ld = np.floor([i/n+1,j/n]).astype(int)
            rd = np.floor([i/n+1,j/n+1]).astype(int)
            if ru[1] >= width:
                ru[1] = width - 1
            if ld[0] >= height:
                ld[0] = height - 1
            if rd[0] >= height:
                rd[0] = height - 1
            if rd[1] >= width:
                rd[1] = width - 1
            if lu[0] <0 or lu[1] <0 or ru[0] <0 or ru[1] >=width or ld[0] >= height or ld[1] <0 or rd[0] >= height or rd[1] >= width:
                raise Exception("fuck up")
            coe_rd, coe_ld, coe_ru, coe_lu = get_coef(lu,ld,ru,rd,i,j,n)
            res[i,j] = coe_rd*img[rd[0],rd[1]] + coe_ld*img[ld[0],ld[1]] + coe_ru*img[ru[0],ru[1]] + coe_lu*img[lu[0],lu[1]]
    return res

def main():      
    # Load the image and points
    img = cv2.imread("./bg.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    points = np.loadtxt("./points.txt")
    
    ##You shold modify result1 , result2 , result
    ## 1.a
    low_detail_points = bezier_curve(points,'low_detail')
    high_detail_points = bezier_curve(points,'high_detail')
    result1 = low_detail_points
    result2 = high_detail_points
    subplot(points  , result1 , result2 , img)
    # 2.a
    scale = 4
    scaled_points = points*scale
    high_detail_points = bezier_curve(scaled_points,'high_detail')
    result = high_detail_points
    img_4x = bilinear_interpolation(img, scale)
    plot(scaled_points  , result , img_4x)

if __name__ == "__main__":
    is_out_exist = os.path.exists('output')
    if not is_out_exist:
        # Create a new directory because it does not exist
        os.makedirs('output')
        print("The new directory is created!")
    main()