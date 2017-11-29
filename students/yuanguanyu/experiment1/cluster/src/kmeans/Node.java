package kmeans;

public class Node {
	double x;
	double y;
	int lable;
	boolean visit;
	boolean isNoised;
	public Node(double x, double y) {
		// TODO Auto-generated constructor stub
		this.x = x;
		this.y = y;
		lable = 0;
		visit = false;
		isNoised = false;
	}
	void setlable(int lable){
		this.lable = lable;
	}
}
