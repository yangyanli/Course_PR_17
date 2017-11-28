package DBScan;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class Data {
    public static ArrayList<Point> getData(String sourcePath) {
        ArrayList<Point> points = new ArrayList<Point>();
        File fileIn = new File(sourcePath);
        try {
            BufferedReader br = new BufferedReader(new FileReader(fileIn));
            String line = null;
            line = br.readLine();
            int count = 0;
            while (line != null) {
                Double x = Double.parseDouble(line.split(" ")[0]);
                Double y = Double.parseDouble(line.split(" ")[1]);
                points.add(new Point(x, y));
                line = br.readLine();
                count++;
            }
            br.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return points;
    }
    public static void writeData(ArrayList<Point> points,String path) {
        try {
            BufferedWriter bw = new BufferedWriter(new FileWriter(path));
            for (Point point:points) {
                bw.write(point.toString()+"\r\n");
            }
            bw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
