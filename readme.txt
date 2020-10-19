~~~ PCA-kMeans-v0/readme.txt ~~~

This is the our first attempt at using the PCA & kmeans methods to cluster our data. It contains 6 python files, five of which contain their own class. The basic function of each file and it's dependencies are outlined below.

~~~ Stat.py ~~~
This class basically handles the re-scaling of data so that it can be better interpreted by other classes.
~~~ Timer.py ~~~
This class is just a basic timer for other classes to help the user see what calculations are taking how long.
~~~ ImPCA.py ~~~
This class does the PCA analysis on image data, and it relies on Stat and Timer classes.
~~~ kMeans.p y~~~
This class does the kmeans clustering of the new data produced by the ImPCA class. It only relies on the Stat class.
~~~ ImageHelper.py ~~~
Since, more often than not, one must alter the raw image data before feeding it through this process, the ImageHelper class takes in the raw image data and the indecies of the elements in each cluster from the kMeans class, and outputs the clusters to a specified folder.
~~~ main.py ~~~
This is the main file, running this file applies the PCA-kMeans to the data you point it to. Variables can be altered under the #VARIABLES comment at the top of the file.
