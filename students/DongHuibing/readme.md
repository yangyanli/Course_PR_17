# Intro
The projects contains implementations of several clustering algorithms, they are:
* [DBSCAN](DBSCAN\src\DBSCAN.cpp)
* kMeans
	* [farthest distance initialization](kMeans-farthestInitMeans\src\kMeans-farthestInitMeans.cpp)
	* [random initialization](kMeans-randomInitMeans\src\kMeans-randomInitMeans.cpp)
* [Spectral Clustering](SpectralClustering\src\spectral.py)

# Usage
1. Original data should be added in `data\synthetic_data\`
2. If a cpp file and a html is included in an implentation, you need to run the cpp to generate data first, and visualization is avaliable using the html file(note that the html file should be opend with Edge, or you need to build a server).