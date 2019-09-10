import cv2
import numpy as np
import matplotlib.pyplot as plt

color_scale = {(40,39,104):0,
				(43,63,126):1,
				(56,111,176):2,
				(78,129,195):3,
				(110,150,203):4,
				(124,170,219):5,
				(146,188,227):6,
				(169,207,236):7,
				(184,224,248):8,
				(212,234,245):9,
				(230,243,252):10,
				(243,251,252):11,
				(245,245,245):12,
				(214,214,214):13,
				(190,190,190):14,
				(160,160,160):15,
				(132,132,132):16,
				(95,95,95):17,
				(57,47,47):18,
				(51,40,38):19} 	#Maps RGB values to topography height

color_2_height_map = {0:-9500,
					 1:-8500,
					 2:-7500,
					 3:-6500,
					 4:-5500,
					 5:-4500,
					 6:-3500,
					 7:-2500,
					 8:-1500,
					 9:-500,
					 10:500,
					 11:1500,
					 12:2500,
					 13:3500,
					 14:4500,
					 15:5500,
					 16:6500,
					 17:7500,
					 18:8500,
					 19:9500
					 }			#Maps index of color scale to height

def replace_with_dict2(ar, dic):
    # Extract out keys and values
    k = np.array(list(dic.keys()))
    v = np.array(list(dic.values()))

    # Get argsort indices
    sidx = k.argsort()

    ks = k[sidx]
    vs = v[sidx]
    return vs[np.searchsorted(ks,ar)]

def get_distance_index(img,color_scale):
	for i,key in enumerate(color_scale):
		r = np.full((img.shape[0],img.shape[1]), key[0])
		g = np.full((img.shape[0],img.shape[1]), key[1])
		b = np.full((img.shape[0],img.shape[1]), key[2])
		r = np.dstack([r,g])
		r = np.dstack([r,b])
		curr_dist = np.linalg.norm(r-img, axis=-1)
		if(i==0):
			dist = curr_dist
		else:
			dist = np.dstack([dist,curr_dist])

	result = np.argmin(dist, axis=2)
	return result

def heatmap2d(arr: np.ndarray):
    plt.imshow(arr, cmap='viridis')
    plt.colorbar()
    plt.show()
		
def main():
	#Load an color image in color
    img = cv2.imread('/Users/harsh/Desktop/CMU_Sem_3/MRSD Project II/Real_Project_Work/LRO_data_to_Map/ezgif.com-crop.png',1)
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    # img = cv2.medianBlur(img,3)
    # img = cv2.GaussianBlur(img,(7,7),0)
    indexed_image = get_distance_index(img,color_scale)
    moon_height_map = replace_with_dict2(indexed_image,color_2_height_map)
    print(type(moon_height_map))
    print(moon_height_map)
    heatmap2d(moon_height_map)
    # cv2.imshow('image',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

'''
# Use some form of blur to get rid of the criss cross grid
# See how 1mm = 10km has to be utilised
'''


if __name__ == "__main__": main()