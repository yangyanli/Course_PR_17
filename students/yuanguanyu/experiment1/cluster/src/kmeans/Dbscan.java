package kmeans;

import java.awt.Color;
import java.awt.Graphics;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

import javax.swing.JFrame;
import javax.swing.JPanel;

import kmeans.Kmeans.draw;

public class Dbscan extends JFrame {
	int minPts = 4;
	int mul = 15;
	double r = 4 * mul;
	int max_count = 100;
	int length;
	ArrayList<Node> nodeStore;
	ArrayList<Color> color;

	public Dbscan() {
		super();
		nodeStore = new ArrayList();

		this.setSize(2000, 1200);// 设置窗体的大小
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);// 设置窗体的关闭方式
		color = new ArrayList<>();
		color.add(Color.red);
		color.add(Color.blue);
		color.add(Color.GRAY);
		color.add(Color.GREEN);
		color.add(Color.yellow);
		color.add(Color.cyan);
		color.add(Color.ORANGE);
		color.add(Color.pink);
		color.add(Color.BLACK);
		color.add(Color.magenta);
		draw dw = new draw();
		dw.setSize(2000, 1200);
		this.add(dw);

		this.setTitle("dbscan");// 设置窗体标题

	}

	class draw extends JPanel {
		public void paint(Graphics g) {
			// ----------------------------------------获取数据------------------------------------------------
			try {
				BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(
						"C:\\Users\\冠宇\\Desktop\\Course_PR_17-master\\experiment1\\data\\synthetic_data\\mix.txt")));
				String temp;
				length = 0;
				while ((temp = br.readLine()) != null) {
					String t[] = temp.split(",");
					nodeStore.add(new Node(Double.parseDouble(t[0]) * mul, Double.parseDouble(t[1]) * mul));
					length++;
				}
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			} catch (NumberFormatException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

			int cluster = 1;
			for (int i = 0; i < nodeStore.size(); i++) {
				Node node = nodeStore.get(i);
				if (!node.visit) {
					node.visit = true;
					ArrayList<Node> adjacentNodes = getAdjacentNodes(node, nodeStore);
					if (adjacentNodes != null && adjacentNodes.size() < minPts) {
						node.isNoised = true;
					} else {
						node.lable = cluster;
						for (int j = 0; j < adjacentNodes.size(); j++) {
							Node adjacentNode = adjacentNodes.get(j);
							if (!adjacentNode.visit) {
								adjacentNode.visit = true;
								ArrayList<Node> adjacemtAdjaventNodes = getAdjacentNodes(adjacentNode, nodeStore);
								if (adjacemtAdjaventNodes != null && adjacemtAdjaventNodes.size() >= minPts) {
									adjacentNodes.addAll(adjacemtAdjaventNodes);
								}
							}
							if (adjacentNode.lable == 0) {
								adjacentNode.lable = cluster;
								if (adjacentNode.isNoised) {
									adjacentNode.isNoised = false;
								}
							}
						}
						cluster++;
					}
				}
			}

			super.paint(g);
			for (int i = 0; i < nodeStore.size(); i++) {
				System.out.println(nodeStore.get(i).x);
				g.setColor(color.get(nodeStore.get(i).lable));
				g.fillOval((int) nodeStore.get(i).x, (int) nodeStore.get(i).y, 5, 5);
			}
		}

		ArrayList<Node> getAdjacentNodes(Node centerNode, ArrayList<Node> nodes) {
			ArrayList<Node> adjacentNodes = new ArrayList<Node>();
			for (Node n : nodes) {
				double distance = Math.sqrt(
						(n.x - centerNode.x) * (n.x - centerNode.x) + (n.y - centerNode.y) * (n.y - centerNode.y));
				if (distance < r) {
					adjacentNodes.add(n);
				}
			}
			return adjacentNodes;
		}

	}
}
