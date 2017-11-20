package DBScan;

import java.util.ArrayList;

public class DBScan {
    private double radius;
    private int minPts;
    
    /**
     * 
     * @param radius 半径
     * @param minPts 最小点数
     */
    public DBScan(double radius,int minPts) {
        this.radius = radius;
        this.minPts = minPts;
    }
/**
 * 首先如果自己r范围内有大于minPts个点的话，就将自己分为核心点。
 * 然后看自己包含的点的r范围内是否还有核心点，将其也加入到自己的包含点中
 * @param points
 */
    public void process(ArrayList<Point> points) {
        int size = points.size();
        int idx = 0;
        int cluster = 1;
        while (idx<size) {
            Point p = points.get(idx++);
            if (!p.getVisit()) {
                p.setVisit(true);//set visited
                ArrayList<Point> adjacentPoints = getAdjacentPoints(p, points);
                if (adjacentPoints != null && adjacentPoints.size() < minPts) {
                    p.setNoised(true);
                } else {
                    p.setCluster(cluster);
                    for (int i = 0; i < adjacentPoints.size(); i++) {
                        Point adjacentPoint = adjacentPoints.get(i);
                        if (!adjacentPoint.getVisit()) {
                            adjacentPoint.setVisit(true);
                            ArrayList<Point> adjacentAdjacentPoints = getAdjacentPoints(adjacentPoint, points);
                            if (adjacentAdjacentPoints != null && adjacentAdjacentPoints.size() >= minPts) {
                                for (Point pp : adjacentAdjacentPoints){
                                    if (!adjacentPoints.contains(pp)){
                                        adjacentPoints.add(pp);
                                    }
                                }
                            }
                        }
                        if (adjacentPoint.getCluster() == 0) {
                            adjacentPoint.setCluster(cluster);
                            if (adjacentPoint.getNoised()) {
                                adjacentPoint.setNoised(false);
                            }
                        }
                    }
                    cluster++;
                }
            }
        }
    }
/**
 * 得到自己为核心点的所有点
 * @param centerPoint
 * @param points
 * @return Adjacent
 */
    private ArrayList<Point> getAdjacentPoints(Point centerPoint,ArrayList<Point> points) {
        ArrayList<Point> adjacentPoints = new ArrayList<Point>();
        for (Point p:points) {
            double distance = centerPoint.getDistance(p);
            if (distance<=radius) {
                adjacentPoints.add(p);
            }
        }
        return adjacentPoints;
    }

}
