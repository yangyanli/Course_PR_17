### introduction
* Exp1/FourDataset is the first dataset of experiment1.
  *DB.py is an implementation of DBSCAN. The parameter of different dataset is listed by comments in this program.
  *Kmeansandkmeansplusp.py is an implementation of kmeans&kmeans++
  *SpectralClustering.py is my implementatio of spectral clustering.  **But** i do not know why it does not work.

* Exp1/FourDataset/picture is the Intuitive clustering results of some clustering algorithm.
  * DB+num means the DBSCAN algorithm result in dataset+num
  * Figure_num is the initial picture of dataset.
  * kmeans is kmeans algorithm
  * Kp is kmeans++
  * Sp is spectral clustering (implemented by sklearn)
  
* Exp1/Mnist is the mnist dataset of experiment1. 
  *meanPixelClustering.py is an naive clustering algorithm. The algorithm is based on the fact **that different digit has differnet   total number of the pixel value.** The procedure of this algorithm is:
    *calculate the total pixel value of the pictures.
    *use one clustering algorithm to cluster those values.(Here, i use kmeans (implemented by sklearn))
  *meanPixelClustering.py is an enhanced algorithm. The algorithm is based on the fact **that different digit has differne distribution. Initially, i want to implememt an algorithm how to value the similarity of two distribution. But i do not have much knowledge so it is. ** The procedure of this algorithm is:
    * calculate the total number of pixel value in each colomn,so we can get a 1*28 vector.
    * clustering those vectors by kmeans
    * VisualofMnist.py is visualization tool of Mnist.
    * picture is the results of those two algorithm.
  
  
