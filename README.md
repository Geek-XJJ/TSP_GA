# TSP_GA
基于遗传算法优化TSP问题。Optimization of TSP problem based on genetic algorithm
本代码主要对遗传算法优化TSP问题进行理论的实现。首先定义20个城市，结构为[城市序号，x坐标，y坐标]，然后生成一组初始路线，计算这些路线的欧氏距离并从中挑选出优秀个体进行信息的交叉并产生新的个体，再进行变异操作。最终迭代多次后得到近似最优路线。当然，城市的数量以及具体的x,y坐标都可以自行更改，只需对代码相应部分进行改动即可，后期也还可以做一个数据可视化来显示具体的路线以解决实际问题。
This code is mainly to genetic algorithm optimization TSP problem theoretical realization. First of all, 20 cities are defined, and the structure is [city serial number, x coordinate, y coordinate], then a set of initial routes are generated, Euclidean distances of these routes are calculated, excellent individuals are selected from them for information crossing and new individuals are generated, and then variation operations are carried out. Finally, the approximate optimal route is obtained after several iterations. Of course, the number of cities and the specific x and y coordinates can be changed by themselves, only the corresponding part of the code can be changed, and later you can also do a data visualization to show the specific route to solve practical problems.
