from matplotlib import pyplot as plt
from imzmlparser import _bisect_spectrum
from scipy.stats import pearsonr
from mykmeans import MiniBatchKMeans
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import cv2
from sklearn.cluster import KMeans, DBSCAN
import random
import xlrd
import os

# transform image given X shift, Y shift, angle of rotation (radians) and scaling factor
def transform_image(X,theta,shiftX,shiftY,scale): 
    (height, width) = X.shape[:2]
    d = (height*height*0.25+width*width*0.25)**0.5
    beta = np.arctan(height/width)
    pad_size = np.abs(int(d*np.cos(beta - theta)-width/2))
    vertical_pad = np.zeros((height,pad_size))
    horizontal_pad = np.zeros((pad_size,2*pad_size+width))
    X_pad = np.concatenate((vertical_pad,X,vertical_pad),axis = 1)
    X_pad2 = np.concatenate((horizontal_pad,X_pad,horizontal_pad),axis = 0)
    (height2, width2) = X_pad2.shape[:2]
    # rotation
    M = cv2.getRotationMatrix2D((width2/2,height2/2), np.degrees(theta), 1)
    Y = cv2.warpAffine(X_pad2, M, (width2,height2))
    # scaling
    Y = cv2.resize(Y, dsize=(int(width2*scale),int(height2*scale)))
    (height3, width3) = Y.shape[:2]
    padY = int((height3-height)/2)
    padX = int((width3-width)/2)
    # shift
    if height3 < height and width3<width:
        output = np.zeros((height, width))
        output[-padY:-padY+height3,-padX:-padX+width3] = Y
        M = np.float32([[1, 0, shiftX], [0, 1, shiftY]])
        res = cv2.warpAffine(output, M, (width,height))
        return res
    elif height3 > height and width3 > width:
        M = np.float32([[1, 0, shiftX], [0, 1, shiftY]])
        res = cv2.warpAffine(Y, M, (width3,height3))
        return res[padY:padY+height,padX:padX+width]
    else:
        M = np.float32([[1, 0, shiftX], [0, 1, shiftY]])
        res = cv2.warpAffine(Y, M, (width3,height3))
        return cv2.resize(res, dsize=(width,height)) 
    
def mutual_information(p,X,Y):
    (height, width) = X.shape[:2]
    if int(p[3]*height)*int(p[3]*width) == 0: return 0
    transformed_Y = transform_image(Y,p[0],p[1],p[2],p[3])
    [top_maldi,bottom_maldi,left_maldi,right_maldi] = crop(transformed_Y)
    [top_confocal,bottom_confocal,left_confocal,right_confocal] = crop(X)
    top,bottom = max(top_maldi,top_confocal),min(bottom_maldi,bottom_confocal)
    left,right = max(left_maldi,left_confocal),min(right_maldi,right_confocal)
    transformed_Y = transformed_Y[top:bottom,left:right]
    X = X[top:bottom,left:right]
    # add a fine factor to account for the parts of the image that were placed outside of the initial field of view
    fine = np.sum(transformed_Y > 0)/np.sum(Y > 0)
    X = X.ravel()
    transformed_Y = transformed_Y.ravel()
    # calculating mutual information
    hgram, _, _ = np.histogram2d(X,transformed_Y,bins=20)
    pxy = hgram / float(np.sum(hgram))
    py = np.sum(pxy, axis=1) 
    px = np.sum(pxy, axis=0) 
    px_py = py[:, None] * px[None, :] 
    nzs = pxy > 0 

    return -fine*fine*np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def crop(image):
    # crop out stripes of zeros on the sides of the image
    shape = image.shape
    top,bottom,left,right = 0,shape[0],0,shape[1]
    for i in range(shape[0]):
        s = np.sum(image[i,:])
        if s > 0:
            bottom = i
            if top == 0: top = i
    for i in range(shape[1]):
        s = np.sum(image[:,i])
        if s > 0:
            right = i
            if left == 0: left = i  
    return top,bottom,left,right

def cleanup(cellprops, h, w, output_folder):
    n = len(cellprops)
    area = np.zeros(n)
    for i in range(n): 
        area[i] = cellprops[i]["area"]
    area_std, area_mean = np.std(area), np.mean(area)
    cells, X, Y, area, perimeter, eccentricity = [],[],[],[],[],[]
    for i in range(n):
        if cellprops[i]['area'] < area_mean+3*area_std and cellprops[i]['area'] > max(5,area_mean-3*area_std):
            cells.append(cellprops[i])
            X.append(cellprops[i]['centroid'][0])
            Y.append(cellprops[i]['centroid'][1])
            area.append(cellprops[i]["area"])
            perimeter.append(cellprops[i]["perimeter"])
            eccentricity.append(cellprops[i]["eccentricity"])                
                
    return cells, np.array(X), np.array(Y), area, perimeter, eccentricity

def relabel_random(labels):
    m = np.max(labels)
    random_labels = np.linspace(1,m,m)
    random.shuffle(random_labels)
    for label in range(1,m):
        labels[np.where(labels==label)] = random_labels[label-1]
    return labels

def plot(image,name):
    image_std, image_mean = np.std(image), np.mean(image)
    image[np.where(image>(image_mean+2*image_std))] = image_mean+2*image_std
    colors_list = [(1, 1, 1), (0, 0, 1), (0, 1, 0), (1,1,0),(1, 0, 0)]
    cm = LinearSegmentedColormap.from_list('cmap_name', colors_list, N=100)
    image = plt.imshow(image, cmap=cm)
    plt.colorbar()
    plt.savefig(name,dpi=1200,bbox_inches = "tight")
    plt.close()

def overlay_cells(image, cells, name,dapi=None):
    h,w = image.shape
    output = np.zeros((h,w))
    n = len(cells)
    intensities = np.zeros(n)
    for i in range(n):
        for coord in cells[i]["coords"]:
            intensities[i] += image[coord[0],coord[1]]
        intensities[i] = intensities[i]/cells[i]["area"]
        for coord in cells[i]["coords"]:
            output[coord[0],coord[1]] = intensities[i]
    if dapi is not None:
        output[np.where(dapi!=0)] = output[np.where(dapi!=0)]/dapi[np.where(dapi!=0)]
    plot(output,name)
    return intensities,output

# visualize metrics, saving in the output folder provided through the "name" variable
def visualize(metric, cells, h, w, name):
    output = np.zeros((h,w))
    n = len(cells)
    for i in range(n):
        for coord in cells[i]["coords"]:
            output[coord[0],coord[1]] = metric[i]
    plot(output,name)

# get average distance between cells' centers from a random rectangular sample in the image
def get_connection_length(X,Y):
    left = random.randrange(int(np.min(X)), int(0.75*np.max(X)))
    top = random.randrange(int(np.min(Y)), int(0.75*np.max(Y)))
    rand_ind = np.where(np.logical_and(np.logical_and(X > left,X < (left + 0.25*np.max(X))), np.logical_and(Y > top, Y < (top + 0.25*np.max(Y)))))
    X = X[rand_ind]
    Y = Y[rand_ind]
    n = len(X)
    connections = np.zeros(n)
    for i in range(n):
        d = np.max(X)
        for j in range(n):
            if i == j: continue
            dist = ((X[j]-X[i])**2+(Y[j]-Y[i])**2)**0.5
            if dist < d: d = dist
        connections[i] = d
    mu = np.mean(connections)
    sigma = 3*np.std(connections)
    main_body = np.logical_and(connections < (mu+sigma), connections > (mu-sigma))
    connections = connections[main_body]
    l = np.max(connections)
    return l

# identify if the cell is on the edge of the colony
def on_edge(X,Y,i,count,neigh,margin):
    selfX = X[i]
    selfY = Y[i]
    # cells closer to the edge of the image than the given margin are considered "on-colony"
    if selfX < margin or selfY < margin or selfX > max(X)-margin or selfY > max(Y)-margin: return 0
    # output 0 means cell is off-edge, 1 - on the edge
    for j in range(count):
        neighX = X[neigh[j]]
        neighY = Y[neigh[j]]
        if selfX == neighX:
            flag = 1
        else:
            a = (selfY - neighY) / (selfX-neighX)
            b = selfY - a * selfX
            flag = 0
        s = 0
        for k in range(count):
            if k == j: continue
            if flag == 0:
                s = s + np.sign(Y[neigh[k]]-a*X[neigh[k]]-b) 
            else:
                s = s + np.sign(Y[neigh[k]] - selfY)
        if np.abs(s) == count: return 1
        else: edge = 0
            
    return edge

# calculating distance from the edge of the colony for a given cell
def edgeDistance(X,Y,edge,i):
    distances = ((X[np.where(edge==1)]-X[i])**2+(Y[np.where(edge==1)]-Y[i])**2)**0.5
    return np.min(distances)

def ClusterImage(Z,k,X,Y,epsilon,minclust):
    if np.sum(Z) == 0: return np.zeros(len(Z)), np.ones(len(Z))
    kmeans = KMeans(n_clusters=k, random_state=0).fit(Z.reshape(-1, 1))
    idx = kmeans.labels_
    l = len(Z)
    cluster_sizes = np.zeros(l)
    cluster_avg = np.zeros(l)
    nums = np.arange(0,l)
    randomness = 0
    for i in range(k):
        indices = nums[np.where(idx == i)]
        x = X[indices]
        y = Y[indices]
        z = Z[indices]
        coeffs = nums[indices]
        clustering = DBSCAN(eps=epsilon, min_samples=minclust).fit(np.array([x,y]).reshape(-1, 2))
        idx2 = clustering.labels_
        dump = 0
        sizes = []
        avgs = []
        for j in range(len(idx2)):
            if idx2[j] == -1:
                sizes.append(1)
                if z[j] == 0:
                    avgs.append(0.01)
                else:
                    avgs.append(z[j])
                dump += 1
            else:
                sizes.append(np.sum(idx2 == idx2[j])) 
                if np.sum(z[np.where(idx2 == idx2[j])]) == 0:
                    avgs.append(0.01)
                else:
                    avgs.append(np.sum(z[np.where(idx2 == idx2[j])])/np.sum(idx2 == idx2[j]))
        cluster_sizes[coeffs] = sizes
        cluster_avg[coeffs] = avgs
        randomness = randomness + np.max(idx2) + 2*dump
        nums[np.where(idx == i)] = idx2+np.max(nums)*np.ones(len(idx2))-np.array(idx2==0,dtype = int)*np.max(nums)
    
    randomness = randomness/len(Z)

    return cluster_sizes, cluster_avg

def getmaxmin(p):
    xmax, xmin, ymax, ymin = 0,600000,0,600000
    for i, (x, y, z) in enumerate(p.coordinates):
        if x>xmax:xmax = x
        if x<xmin:xmin = x
        if y>ymax:ymax = y
        if y<ymin:ymin = y
    return xmax, xmin, ymax, ymin

def getimagedimsfromexcel(folder):
    wb = xlrd.open_workbook(os.path.join(folder,"raw pixel data.xlsx"))
    sheet = wb.sheet_by_index(0)
    ydim,xdim = 0,0
    for i in range(1,sheet.nrows):
        if sheet.cell_value(i, 1) > ydim: ydim = sheet.cell_value(i, 1)
        if sheet.cell_value(i, 0) > xdim: xdim = sheet.cell_value(i, 0)
    return ydim,xdim

def readimagefromexcel(mz_value,folder,ydim,xdim):
    wb = xlrd.open_workbook(os.path.join(folder,"raw pixel data.xlsx"))
    sheet = wb.sheet_by_index(0)
    img = np.zeros((int(ydim)+1,int(xdim)+1))
    for i in range(sheet.ncols):
        if sheet.cell_value(0, i) == mz_value:break
    for j in range(1,sheet.nrows):
        if sheet.cell_value(j, sheet.ncols-1) == 0: continue
        img[int(sheet.cell_value(j, 1))-1,int(sheet.cell_value(j, 0))-1] = sheet.cell_value(j, i)
    return img

def getionimage(borders,p, mz_values, tolerances, z=1, reduce_func=sum,dim=1,):
    coordsX = []
    coordsY = []
    im = np.zeros((dim, borders[2]-borders[1]+1,  borders[3]-borders[0]+1))
    sum_im = np.zeros((borders[2]-borders[1]+1,  borders[3]-borders[0]+1))
    for i, (x, y, z_) in enumerate(p.coordinates):
        if z_ == z and x >= borders[0] and y >= borders[1] and x <= borders[3] and y <= borders[2]:
            mzs, ints = p.getspectrum(i)
            sum_im[y-borders[1]-1, x-borders[0]-1] = np.sum(ints)
            coordsX.append(x-borders[0])
            coordsY.append(y-borders[1])

            for index,mz_value in enumerate(mz_values):
                min_i, max_i = _bisect_spectrum(mzs, mz_value, tolerances[index])
                integral_signal = reduce_func(ints[min_i:max_i+1])
                im[index,y-borders[1]-1, x-borders[0]-1] = integral_signal

    return im, sum_im, coordsX, coordsY

def record_reader(borders,p,MALDI_output,mz_values,tolerances):
    img,sum_im,coordsX,coordsY = getionimage(borders,p, mz_values, tolerances, z=1, reduce_func=sum,dim=len(mz_values))
    average = np.sum(img,axis=0)/len(mz_values)
    plt.imsave(os.path.join(MALDI_output,"average.png"), cmap(average/np.max(average)))
    plt.figure(figsize=(10,10))
    plt.imshow(average,cmap="gray")
    plt.colorbar()
    plt.show()
    for index,values in enumerate(img):
        a = cmap(values/np.max(values))
        exten = str(mz_values[index])
        exten.replace(".","_")
        plt.imsave("{}//MALDI__{}.png".format(MALDI_output,exten), a)
    workbook = xlsxwriter.Workbook(os.path.join(MALDI_output,"raw pixel data.xlsx"))
    worksheet = workbook.add_worksheet()
    worksheet.write(0,0,"X")
    worksheet.write(0,1,"Y")    
    worksheet.write_row(0,2,mz_values)
    worksheet.write_column(1,0,coordsX)
    worksheet.write_column(1,1,coordsY)
    for index,values in enumerate(img):
        for i in range(len(coordsX)):
            if sum_im[coordsY[i],coordsX[i]] == 0:worksheet.write(i+1, index+2, 0)
            else:    
                worksheet.write(i+1, index+2, 1000*values[coordsY[i],coordsX[i]]/sum_im[coordsY[i],coordsX[i]])  
    workbook.close()   
