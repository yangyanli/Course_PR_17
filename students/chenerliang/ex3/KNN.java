import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

public class KNN {
	//KNN的关键参数K
	public static int K = 3;
	
	
	public static void main(String args[]){
/*		 //预测正确的数字个数
		 double correctNum = 0;
		 //预测的所有数字个数
		 double allNum = 0;
		 //预测的是否正确标识
		 boolean flag = false;
		 
		 for(int i= 0;i<10;i++ ){//对0~9这10个数字进行预测
			 for(int j=0;j<40;j++){//分别选取测试集中的40个样本进行测试
				 //进行预测
				 int back = predict("C:/Users/hp/Desktop/KNN/测试集/"+i+"_"+j+".txt");
				 //预测数字的个数加1
				 allNum++;
				 //如果预测正确，则预测正确个数加1，flag置为true
				 if(back == i){
					 correctNum++;
					 flag = true;
				 }
				//输出语句
				 if(flag){
					 System.out.println("预测正确，"+"真实值："+i+"，预测值："+back);
					 flag = false;
				 }
				 else{
					 System.out.println("预测错误，"+"真实值："+i+"，预测值："+back); 
				 }
			 }
		 }
		 
		 System.out.println("测试集总数："+allNum+"，正确个数："+correctNum);
		 System.out.println("准确率："+correctNum/allNum);
*/
		
		int predict =  predict("C:/Users/hp/Desktop/KNN/手写字符集/"+"a.txt");
		System.out.println("预测的数字是："+predict);
	}

	//将从文本中读到的32*32个0,1串放到一个数组中
	private static int[] dataToVector(String fileName){
		//创建一个32*32的数组用来存放读入的0,1串
		int array[] = new int[32*32];
		//读取文件内容并进行操作
		try{
            FileReader reader = new FileReader(fileName);
            BufferedReader buffer = new BufferedReader(reader);
            //对文件的32行32列分别进行读取
            for(int row=0; row<32; row++){
                String str = buffer.readLine();
                for(int col=0; col<32; col++){
                    String c = "" + str.charAt(col);
                    array[32*row+col] = Integer.parseInt(c);
                }
            }
        }catch (FileNotFoundException e){
            e.printStackTrace();
        }catch (IOException e){
            e.printStackTrace();
        }
        return array;
	}
	
	//计算两点之间的欧式距离
	public static double computeDistance(int aVector[],int bVector[]){
		//距离的最终结果
		double distance = 0;
		//距离的平方
		int distance2 = 0;
		//计算distance2
		for(int i=0;i<32*32;i++){
			int difference = aVector[i]-bVector[i];
			distance2 = distance2 + difference*difference;
		}
		//距离等于距离的平方开根号
		distance = Math.sqrt(distance2);	
		return distance;
	}
	
	private static int predict(String fileName) {
		//预测值
		int back=-1;
		//获得要预测的32*32向量
		int aVector[] = dataToVector(fileName);
		//与要预测的点最近的K个点及它们的距离
		double distanceArray[] = new double[K];
		int numArray[] = new int[K];
		//初始化
		for(int i=0;i<K;i++){
			distanceArray[i]=32;
			numArray[i]=-1;
		}
		//计算最近点及其距离
		 for(int i = 0; i <= 9; i++){
	            for(int j = 0; j < 100; j++){
	                int bVector[] = dataToVector("C:/Users/hp/Desktop/KNN/训练集/"+i+"_"+j+".txt");
	                double distance = computeDistance(aVector, bVector);

	                for(int k = 0; k < K; k++){
	                    if(distance < distanceArray[k]){
	                    	for(int h=k;h<K-1;h++){
	                    		distanceArray[h+1] = distanceArray[h];
	                    		numArray[h+1] = numArray[h];
	                    	}	                    	
	                        distanceArray[k] = distance;
	                        numArray[k] = i;
	                        break;
	                    }
	                }
	            }
	        }
		//统计最近点中不同数字的个数，数字i的个数存在count[i]中
		int count[] = new int[10];
		//count[]初始化
		for(int i=0;i<10;i++){
			count[i]=0;
		}
		//统计
		for(int i=0;i<K;i++){
			if(numArray[i]!=-1){
				count[numArray[i]]++;
			}
		}
		//计算个数最多的数字i
		int max=0;
		for(int i=0;i<10;i++){
			if(count[i]>max){
				max = count[i];
				back = i;
			}
		}
		
		return back;
	}

	
}