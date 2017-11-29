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
//import javax.swing.JScrollPane;

public class Kmeans extends JFrame {
	int k = 9;
	int mul = 15;
	int max_count = 100;
	int length;
	ArrayList<Node> nodeStore;
	ArrayList<Color> color;

	public Kmeans() {
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

		this.setTitle("kmeans");// 设置窗体标题

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
					nodeStore.add(new Node(Double.parseDouble(t[0])*mul, Double.parseDouble(t[1])*mul));
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

			// -------------------------------迭代---------------------------------
			int count = 0;
			Node[] core = new Node[k]; // 中心点
			ArrayList<Node> lableStore[] = new ArrayList[k];
			for (int i = 0; i < k; i++) {
				core[i] = new Node(Math.random() * 50*mul, Math.random() * 50*mul);
				lableStore[i] = new ArrayList<Node>();
			}
			while (count < max_count) {
				for (int i = 0; i < nodeStore.size(); i++) {
					double belong = 1000;
					int belong_c = 0;
					for (int c = 0; c < k; c++) {
						double dx = nodeStore.get(i).x - core[c].x;
						double dy = nodeStore.get(i).y - core[c].y;
						double distance = Math.sqrt(dx * dx + dy * dy);
						if (distance < belong) {
							belong = distance;
							belong_c = c;
						}
					}
					nodeStore.get(i).lable = belong_c;
					lableStore[belong_c].add(nodeStore.get(i));
				}
				for (int i = 0; i < k; i++) {
					double sum_x = 0;
					double sum_y = 0;
					for (int n = 0; n < lableStore[i].size(); n++) {
						sum_x += lableStore[i].get(n).x;
						sum_y += lableStore[i].get(n).y;
					}
					core[i].x = sum_x / lableStore[i].size();
					core[i].y = sum_y / lableStore[i].size();
					
				}

				count++;
				
			}
			// -------------------------绘图-------------------------------------
			super.paint(g);
			for (int i = 0; i < nodeStore.size(); i++) {
				System.out.println(nodeStore.get(i).x);
				g.setColor(color.get(nodeStore.get(i).lable));
				g.fillOval((int)nodeStore.get(i).x, (int)nodeStore.get(i).y, 5, 5);
			}
		}
	}

}
