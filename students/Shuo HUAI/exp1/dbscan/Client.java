package DBScan;

import java.util.ArrayList;
public class Client {
	public static void main(String[] args) {
		ArrayList<Point> points = Data.getData(path);
		DBScan dbScan = new DBScan(1,6);
		dbScan.process(points);
		for (Point p : points) {
			System.out.println(p);
		}
		Data.writeData(points, Mypath);
	}
	public static String path = "E:\\下载\\Course_PR_17-master\\experiment1\\data\\synthetic_data\\flame.txt";
	public static String Mypath = "E:\\下载\\Course_PR_17-master\\experiment1\\data\\synthetic_data\\dbflame.txt";
	
}
