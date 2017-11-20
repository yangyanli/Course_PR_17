package K_means;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.Scanner;
/**
 * 输入k的个数
 * @author 19478
 *
 */
public class Kcluster {
	public static void main(String[] args) throws IOException {
		int count = 0;
		File f = new File(
				"E:\\下载\\Course_PR_17-master\\experiment1\\data\\synthetic_data\\flame.txt");
		InputStream input = new FileInputStream(f);
		BufferedReader b = new BufferedReader(new InputStreamReader(input));
		String value = b.readLine();
		while (value != null && !value.equals("")) {
			count++;
			value = b.readLine();
		}
		Point[] points = new Point[count];//定义count个点
		count = 0;
		b.close();
		input.close();
		input = new FileInputStream(f);
		b = new BufferedReader(new InputStreamReader(input));
		value = b.readLine();
		while (value != null && !value.equals("")) {//根据文件初始化点
			String[] local = value.split(" ");
			float x = Float.parseFloat(local[0]);
			float y = Float.parseFloat(local[1]);
			points[count] = new Point(x, y);
			count++;
			value = b.readLine();
		}
		b.close();
		input.close();
		Scanner scanner = new Scanner(System.in);
		int k = scanner.nextInt();
		int[] Kp = new int[k];
		Kp = RandomCommon.randomCommon(0, count - 1, k);// create k random
		float[] xs = new float[k];
		float[] ys = new float[k];
		int[] num = new int[k];
		int times = 0;
		while (true) {
			times++;
			for (int i = 0; i < k; i++) {
				xs[i] = 0;
				ys[i] = 0;
				num[i] = 0;
			}
			int Cnum = 0;// 不动的点
			// count个点 K个关键点 Kp关键点的数组，返回指向points的哪个的指针
			for (int i = 0; i < count; i++) {// 到每个选取的点的距离
				float dis = Float.MAX_VALUE;
				int la = 0;
				for (int j = 0; j < k; j++) {
					float dx = points[i].x - points[Kp[j]].x;
					float dy = points[i].y - points[Kp[j]].y;
					float d;
					d = dx * dx + dy * dy;
					if (d < dis) {
						dis = d;
						la = j;// 将Lable置为第几个关键点
					}
				}
				points[i].lable = la;
			}

			for (int i = 0; i < count; i++) {
				xs[points[i].lable] += points[i].x;// 将lable相同的点的x加起来
				ys[points[i].lable] += points[i].y;
				num[points[i].lable] += 1;
			}
			for (int i = 0; i < k; i++) {// 将新的中心映射到离中心最近的点上
				float nx = xs[i] / ((float) num[i]);
				float ny = ys[i] / ((float) num[i]);
				int q = -1;
				float dis = Float.MAX_VALUE;
				for (int j = 0; j < count; j++) {
					float dx = points[j].x - nx;
					float dy = points[j].y - ny;
					float d;
					d = dx * dx + dy * dy;
					if (d < dis) {
						dis = d;
						q = j;// 将Lable置为第几个关键点
					}
				}
				if (Kp[i] == q) {
					Cnum++;
				}
				Kp[i] = q;
			}
			if (Cnum == k||times>100) {
				break;
			}
		}
		String output = "";
		for (int i = 0; i < count; i++) {
			output += points[i].x + " " + points[i].y + " " + points[i].lable
					+ "\r\n";
		}
		BufferedWriter out = new BufferedWriter(
				new OutputStreamWriter(
						new FileOutputStream(
								"E:\\下载\\Course_PR_17-master\\experiment1\\data\\synthetic_data\\kflame.txt")));
		out.write(output);
		out.flush();
		out.close();
	}
}
