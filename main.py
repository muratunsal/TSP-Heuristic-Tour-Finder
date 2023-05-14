import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import time
import warnings
import math
import sys


warnings.filterwarnings("ignore")



visitedCount = 0
cities = {}
arrayOfCityIDs = []
cityX = []
cityY = []
coordinatesOfClusterCenters = {}


# Read input
with open(sys.argv[1],"r") as f:
    while True:
        line = f.readline()
        if line == '':
            break
        else:
            id, x, y = line.split()
            
            id = int(id)
            x  = int(x)
            y  = int(y)
            
            arrayOfCityIDs.append(int(id))
            cityX.append(x)
            cityY.append(y)
            
            # Add cities to dictionary with its coordinates     
            cities[id] = [x,y]
        
f.close()

numberOfCities = len(cities)



# A Function that checks whether given point is in given square area
def checkIfInSquare(bl, tr, p) :
   if (p[0] > bl[0] and p[0] < tr[0] and p[1] > bl[1] and p[1] < tr[1]) :
      return True
   else :
      return False
        
# A Function that finds "Key" of dictionary from its "Value"
def get_key(val):
   
    for key, value in cities.items():
        if val == value:
            return key

# A Function that finds the distance between two points
def totalDistance(p1,p2):
    distance = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
    return distance


data = list(zip(cityX, cityY))
kmeans = KMeans(n_clusters=2)
kmeans.fit(data)
labels = kmeans.labels_
clusterCount =  np.bincount(labels)




cluster_centers = kmeans.cluster_centers_
cs_x = cluster_centers[:,0]
cs_y = cluster_centers[:,1]


for i in range(len(kmeans.cluster_centers_)):
    coordinatesOfClusterCenters[i] = [kmeans.cluster_centers_[i][0],kmeans.cluster_centers_[i][1]]


for i in range(len(kmeans.cluster_centers_)):
    coordinatesOfClusterCenters[i] = [kmeans.cluster_centers_[i][0],kmeans.cluster_centers_[i][1]]


# midpoint of 2 clusters without weight calculation
sumX = 0
sumY = 0
for coordinate in coordinatesOfClusterCenters:
    sumX = sumX + coordinatesOfClusterCenters[coordinate][0]
    finalX = sumX / len(coordinatesOfClusterCenters)
    sumY = sumY + coordinatesOfClusterCenters[coordinate][1]
    finalY = sumY / len(coordinatesOfClusterCenters)

# A diagram that shows the dataset and midpoint of clusters (without weight) *OPTIONAL*
plt.scatter(finalX, finalY, marker='*', s=150, c='r')
plt.scatter(cityX, cityY, c=kmeans.labels_)
plt.title('KMeans')

#number of points in each cluster
clusterCount =  np.bincount(labels)

# Weights of clusters according to points that they have
cluster1weight = clusterCount[0]/(clusterCount[0]+clusterCount[1])
cluster2weight = 1-cluster1weight

# midpoint of 2 clusters with weight calculation
finalX2 = cluster1weight*coordinatesOfClusterCenters[0][0] + cluster2weight*coordinatesOfClusterCenters[1][0]
finalY2 = cluster1weight*coordinatesOfClusterCenters[0][1] + cluster2weight*coordinatesOfClusterCenters[1][1]

# A diagram that shows the dataset and midpoint of clusters (with weights) *OPTIONAL*
plt.scatter(finalX2, finalY2, marker='*', s=150, c='g')
plt.scatter(cityX, cityY, c=kmeans.labels_)
plt.title('KMeans')

plt.show() 


increment = 0.1  # Increment value for adjusting the square size
nextPoint = None  # Placeholder for the next point
visitedCities = []  # List to store visited cities
listToWriteOntoOutput = []  # List to write output data
startTime = time.time()  # Record the starting time

def solve(arrayOfPoints, point):
    x, y = point
    indexOfClosestOne = -1  # Initialize the index of the closest point
    smallest = float("inf")  # Initialize the smallest distance as infinity

    # Iterate through the points in the array
    for p in arrayOfPoints:
        if p[0] == x or p[1] == y:
            dist = abs(x - p[0]) + abs(y - p[1])  # Calculate the Manhattan distance from the given point

            # Check if the current distance is smaller than the smallest distance found so far
            if dist < smallest:
                indexOfClosestOne = arrayOfPoints.index(p)  # Update the index of the closest point
                smallest = dist  # Update the smallest distance
            # If the distances are equal, prioritize the point with the smaller index
            elif dist == smallest:
                if arrayOfPoints.index(p) < indexOfClosestOne:
                    indexOfClosestOne = arrayOfPoints.index(p)  # Update the index of the closest point
                    smallest = dist  # Update the smallest distance

    return indexOfClosestOne  # Return the index of the closest point


#  ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓-------MAIN ALGORITHM-------↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
pointsInSquare = []  # List to store points within the square

while (nextPoint == None):
    bottom_left = (finalX2 - increment, finalY2 - increment)  # Bottom left coordinate of the square
    top_right = (finalX2 + increment, finalY2 + increment)  # Top right coordinate of the square

    for i in data:
        if (checkIfInSquare(bottom_left, top_right, i)):
            pointsInSquare.append(i)  # Add points within the square to the list

    if len(pointsInSquare) == 0:
        increment += 1  # Increase the square size if no points are found
        continue
    else:
        greenStar = [finalX2, finalY2]
        indexOfClosestOne = solve(pointsInSquare, greenStar)
        point = pointsInSquare[indexOfClosestOne]
        value = [point[0], point[1]]
        listToWriteOntoOutput.append(str(get_key(value)))
        nextPoint = value
        data.remove(point)
        visitedCount += 1
        visitedCities.append(value)
        pointsInSquare.remove(point)
        increment = 0.1  # Reset the increment size


while (visitedCount < numberOfCities / 2):
    bottom_left = (nextPoint[0] - increment, nextPoint[1] - increment)
    top_right = (nextPoint[0] + increment, nextPoint[1] + increment)

    for i in data:
        if (checkIfInSquare(bottom_left, top_right, i)):
            pointsInSquare.append(i)  # Add points within the square to the list

    if len(pointsInSquare) == 0:
        increment += 1  # Increase the square size if no points are found
        continue
    else:
        indexOfClosestOne = solve(pointsInSquare, nextPoint)
        point = pointsInSquare[indexOfClosestOne]
        value = [point[0], point[1]]
        listToWriteOntoOutput.append(str(get_key(value)))
        nextPoint = value
        data.remove(point)
        visitedCount += 1
        visitedCities.append(value)
        pointsInSquare.remove(point)
        increment = 0.1  # Reset the increment size

               

#  ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑-------MAIN ALGORITHM-------↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

endTime = time.time()

def distance(a, b):
    
    # Euclidean distance rounded to the nearest integer:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    
    return int(round(math.sqrt(dx * dx + dy * dy)))


distanceSum = 0
for i in range(len(visitedCities)-1):
    distanceSum+= distance(visitedCities[i],visitedCities[i+1])
distanceSum+= distance(visitedCities[0],visitedCities[-1])




# Printing Distance Traveled & Runtime & Number of Visited Cities
print(f"Distance Traveled: {int(distanceSum)} ")
print("--------------------------------------------")
print(f"Runtime: {endTime-startTime} secs")
print("--------------------------------------------")
print(f"Number of Visited Cities : {len(visitedCities)}")
print("--------------------------------------------")

listToWriteOntoOutput.insert(0, str(int(distanceSum)))


# Write Distance Traveled & Visited City IDs Into Output File
with open(sys.argv[2],"w") as f:
    for item in listToWriteOntoOutput:
        # write each item on a new line
        f.write("%s\n" % item)
    print('\nOutput File is Ready!')
f.close()



