package com.shm.test;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;

/**
 * 结合Lucene和weka对iris数据集的再处理
 * @author SHM
 *
 */
public class Test {

	private static final String IRIS_PATH = "./data/";//iris文件所在路径
    private static final int NUM = 10;
    private static final String NEWIRIS_PATH = "./newData/";//新生成的iris文件保存路径

    
    /**
     * 	根据需求得到新的数据集的方法
     * @param instances
     * @return
     */
    private static List<List<String>> generateNewData(Instances instances) {
        List<List<String>> tableData = new ArrayList<>(); //用List集合存储数据集List中存放的还是List，即每一行数据存入List中，在将得到的所有List放入List中保存
        
        //遍历传入的数据集
        for (int i = 0, len = instances.numInstances(); i < len; i++) {
        	//定义一个list保存每一行数据
            List<String> lineData = new ArrayList<>();
            //从下标为1的数据开始遍历，去除数据集中的第一列
            for (int j = 1, attrLen = instances.numAttributes(); j < attrLen; j++) {
            	lineData.add(instances.get(i).toString(j)); //保存得到每一行数据
            }
            tableData.add(lineData); //将每一行数据放入List中，得到新的数据集tableData
        }
        return tableData;
    }

    /**
     * 	根据关键字检索数据集，得到满足条件的数据集的方法
     * @param data
     * @param keywords
     * @return
     * @throws Exception
     */
    private static List<List<String>> searchKeyWord(List<List<String>> data, String... keywords) throws Exception {
    	//创建标准分词器
        Analyzer analyzer = new StandardAnalyzer();
        Directory directory = FSDirectory.open(Paths.get(NEWIRIS_PATH));////索引库存放的路径

        //创建新的索引库
        IndexWriterConfig config = new IndexWriterConfig(analyzer);
        IndexWriter indexWriter = new IndexWriter(directory, config);

        indexWriter.deleteAll(); //清零
        //遍历数据集
        for (List<String> list : data) {
            Document doc = new Document();//创建文档
            //设置文档的各列域属性，并将数据集按列存储，存入对应属性的列中
            doc.add(new Field("sepalwidth", list.get(0), StringField.TYPE_STORED));
            doc.add(new Field("petallength", list.get(1), StringField.TYPE_STORED));
            doc.add(new Field("petalwidth", list.get(2), StringField.TYPE_STORED));
            doc.add(new Field("class", list.get(3), TextField.TYPE_STORED));
            indexWriter.addDocument(doc);//将文档加入索引库
        }
        indexWriter.close();

        DirectoryReader reader = DirectoryReader.open(directory);
        //创建一个搜索searching提供索引
        IndexSearcher indexSearcher = new IndexSearcher(reader);

        //定义一个List结果集
        List<List<String>> results = new ArrayList<>();
        
        for (String keyword : keywords) {
        	//创建queryparser对象；第一个参数默认搜索的域；第二个参数就是分析器对象
            QueryParser queryParser = new QueryParser("class", analyzer);
            // 通过queryParser解析输入（分词），生成query对象
            Query query = queryParser.parse(keyword);
            // 搜索，得到命中的结果（结果中有命中总数）
            ScoreDoc[] scoreDoc = indexSearcher.search(query, 1000).scoreDocs;

            List<List<String>> result = new ArrayList<>();
            
            // 遍历结果scoreDocs,取出文档id对应的文档信息
            if (scoreDoc.length > 0) {
                for (ScoreDoc sd : scoreDoc) {
                	// 根据文档id取存储的文档
                    Document hitDoc = indexSearcher.doc(sd.doc);
                    List<String> lineData = new ArrayList<>();
                    //按属性 取文档的字段
                    lineData.add(hitDoc.get("sepalwidth"));
                    lineData.add(hitDoc.get("petallength"));
                    lineData.add(hitDoc.get("petalwidth"));
                    lineData.add(hitDoc.get("class"));
                    result.add(lineData); //存入List中
                }
                results.addAll(result);
            } else {
                System.out.println("未查询到该关键字");
            }
        }
        reader.close();
        directory.close();

        return results;
    }

    /**
     * 	生成Arff格式的实例数据集
     * @param data
     * @param classes
     * @param attrs
     * @return
     */
    private static Instances generateInstance(List<List<String>> data, ArrayList<String> classes, String... attrs) {
        ArrayList<Attribute> attributes = new ArrayList<>();
        //将4列属性存入数组
        for (int i = 0, len = attrs.length; i < len; i++) {
            if (i != 3) {
                attributes.add(new Attribute(attrs[i]));
            } else {
                attributes.add(new Attribute(attrs[i], classes));
            }
        }

        //设置Arff文件头，包括文件名及属性字段
        Instances instances = new Instances("new_iris", attributes, 0);
        instances.setClassIndex(instances.numAttributes() - 1);

        //遍历数据集合，存入实例中，形成最终结果集
        for (List<String> list : data) {
            Instance instance = new DenseInstance(attributes.size());
            for (int i = 0, len = attrs.length; i < len; i++) {
                instance.setDataset(instances);
                if (i != 3) {
                    instance.setValue(i, Double.valueOf(list.get(i)));
                } else {
                    instance.setValue(i, list.get(i));
                }
            }
            instances.add(instance);
        }

        return instances;
    }

    private static void generateArffFile(Instances instances, String path) {
        ArffSaver saver = new ArffSaver();
        saver.setInstances(instances);
        try {
            saver.setFile(new File(path));
            saver.writeBatch();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        try {
        	//创建加载器
            ArffLoader loader = new ArffLoader();
            //读入文件
            loader.setFile(new File(IRIS_PATH + "iris.arff"));
            //获取数据集，得到所有实例
            Instances instances = loader.getDataSet();
            //设置类别属性  
            instances.setClassIndex(instances.numAttributes() - 1);

            //处理得到的数据集，生成新的数据集
            List<List<String>> data = generateNewData(instances);
            
            //根据"setosa", "versicolor"关键字检索数据集，得到满足条件的数据集
            List<List<String>> newData = searchKeyWord(data, "setosa", "versicolor");

            ArrayList<String> classList = new ArrayList<>();
            classList.add("Iris-setosa");
            classList.add("Iris-versicolor");

            //生成Arff格式的实例数据集
            Instances newInstances = generateInstance(newData, classList, "sepalwidth", "petallength", "petalwidth", "class");
            newInstances.setClassIndex(newInstances.numAttributes() - 1);

            
            generateArffFile(newInstances, NEWIRIS_PATH + "new_iris.arff");

            System.out.println(newInstances);
            System.out.println("-------------------------------------");

//            RandomForest randomForest = new RandomForest(); //随机森林分类器
//            randomForest.buildClassifier(newInstances);
            
            Classifier classifier = new Logistic(); //逻辑回归分类器
            classifier.buildClassifier(newInstances); //训练

            Evaluation evaluation = new Evaluation(newInstances);
            evaluation.crossValidateModel(classifier, newInstances, NUM, new Random(123));
            System.out.println(evaluation.toSummaryString());
            System.out.println(evaluation.toClassDetailsString());
            System.out.println(evaluation.toMatrixString());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
	

    
    
 
    
    
    
	String content;
	
	/**
	 *	读取文件    
	 * @param file
	 * @throws IOException
	 */
	public void readFile(File file) throws IOException {
		BufferedReader bf = new BufferedReader(new FileReader(file));
		String lineContent = "";
		StringBuilder sb = new StringBuilder();
		while (lineContent != null) {
			lineContent = bf.readLine();
			if (lineContent == null) {
				break;
			}
			sb.append(lineContent).append(" ");
		}
		content = sb.toString();
		
		bf.close();
	}
		
}
