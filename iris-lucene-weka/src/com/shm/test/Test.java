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
 * ���Lucene��weka��iris���ݼ����ٴ���
 * @author SHM
 *
 */
public class Test {

	private static final String IRIS_PATH = "./data/";//iris�ļ�����·��
    private static final int NUM = 10;
    private static final String NEWIRIS_PATH = "./newData/";//�����ɵ�iris�ļ�����·��

    
    /**
     * 	��������õ��µ����ݼ��ķ���
     * @param instances
     * @return
     */
    private static List<List<String>> generateNewData(Instances instances) {
        List<List<String>> tableData = new ArrayList<>(); //��List���ϴ洢���ݼ�List�д�ŵĻ���List����ÿһ�����ݴ���List�У��ڽ��õ�������List����List�б���
        
        //������������ݼ�
        for (int i = 0, len = instances.numInstances(); i < len; i++) {
        	//����һ��list����ÿһ������
            List<String> lineData = new ArrayList<>();
            //���±�Ϊ1�����ݿ�ʼ������ȥ�����ݼ��еĵ�һ��
            for (int j = 1, attrLen = instances.numAttributes(); j < attrLen; j++) {
            	lineData.add(instances.get(i).toString(j)); //����õ�ÿһ������
            }
            tableData.add(lineData); //��ÿһ�����ݷ���List�У��õ��µ����ݼ�tableData
        }
        return tableData;
    }

    /**
     * 	���ݹؼ��ּ������ݼ����õ��������������ݼ��ķ���
     * @param data
     * @param keywords
     * @return
     * @throws Exception
     */
    private static List<List<String>> searchKeyWord(List<List<String>> data, String... keywords) throws Exception {
    	//������׼�ִ���
        Analyzer analyzer = new StandardAnalyzer();
        Directory directory = FSDirectory.open(Paths.get(NEWIRIS_PATH));////�������ŵ�·��

        //�����µ�������
        IndexWriterConfig config = new IndexWriterConfig(analyzer);
        IndexWriter indexWriter = new IndexWriter(directory, config);

        indexWriter.deleteAll(); //����
        //�������ݼ�
        for (List<String> list : data) {
            Document doc = new Document();//�����ĵ�
            //�����ĵ��ĸ��������ԣ��������ݼ����д洢�������Ӧ���Ե�����
            doc.add(new Field("sepalwidth", list.get(0), StringField.TYPE_STORED));
            doc.add(new Field("petallength", list.get(1), StringField.TYPE_STORED));
            doc.add(new Field("petalwidth", list.get(2), StringField.TYPE_STORED));
            doc.add(new Field("class", list.get(3), TextField.TYPE_STORED));
            indexWriter.addDocument(doc);//���ĵ�����������
        }
        indexWriter.close();

        DirectoryReader reader = DirectoryReader.open(directory);
        //����һ������searching�ṩ����
        IndexSearcher indexSearcher = new IndexSearcher(reader);

        //����һ��List�����
        List<List<String>> results = new ArrayList<>();
        
        for (String keyword : keywords) {
        	//����queryparser���󣻵�һ������Ĭ���������򣻵ڶ����������Ƿ���������
            QueryParser queryParser = new QueryParser("class", analyzer);
            // ͨ��queryParser�������루�ִʣ�������query����
            Query query = queryParser.parse(keyword);
            // �������õ����еĽ���������������������
            ScoreDoc[] scoreDoc = indexSearcher.search(query, 1000).scoreDocs;

            List<List<String>> result = new ArrayList<>();
            
            // �������scoreDocs,ȡ���ĵ�id��Ӧ���ĵ���Ϣ
            if (scoreDoc.length > 0) {
                for (ScoreDoc sd : scoreDoc) {
                	// �����ĵ�idȡ�洢���ĵ�
                    Document hitDoc = indexSearcher.doc(sd.doc);
                    List<String> lineData = new ArrayList<>();
                    //������ ȡ�ĵ����ֶ�
                    lineData.add(hitDoc.get("sepalwidth"));
                    lineData.add(hitDoc.get("petallength"));
                    lineData.add(hitDoc.get("petalwidth"));
                    lineData.add(hitDoc.get("class"));
                    result.add(lineData); //����List��
                }
                results.addAll(result);
            } else {
                System.out.println("δ��ѯ���ùؼ���");
            }
        }
        reader.close();
        directory.close();

        return results;
    }

    /**
     * 	����Arff��ʽ��ʵ�����ݼ�
     * @param data
     * @param classes
     * @param attrs
     * @return
     */
    private static Instances generateInstance(List<List<String>> data, ArrayList<String> classes, String... attrs) {
        ArrayList<Attribute> attributes = new ArrayList<>();
        //��4�����Դ�������
        for (int i = 0, len = attrs.length; i < len; i++) {
            if (i != 3) {
                attributes.add(new Attribute(attrs[i]));
            } else {
                attributes.add(new Attribute(attrs[i], classes));
            }
        }

        //����Arff�ļ�ͷ�������ļ����������ֶ�
        Instances instances = new Instances("new_iris", attributes, 0);
        instances.setClassIndex(instances.numAttributes() - 1);

        //�������ݼ��ϣ�����ʵ���У��γ����ս����
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
        	//����������
            ArffLoader loader = new ArffLoader();
            //�����ļ�
            loader.setFile(new File(IRIS_PATH + "iris.arff"));
            //��ȡ���ݼ����õ�����ʵ��
            Instances instances = loader.getDataSet();
            //�����������  
            instances.setClassIndex(instances.numAttributes() - 1);

            //����õ������ݼ��������µ����ݼ�
            List<List<String>> data = generateNewData(instances);
            
            //����"setosa", "versicolor"�ؼ��ּ������ݼ����õ��������������ݼ�
            List<List<String>> newData = searchKeyWord(data, "setosa", "versicolor");

            ArrayList<String> classList = new ArrayList<>();
            classList.add("Iris-setosa");
            classList.add("Iris-versicolor");

            //����Arff��ʽ��ʵ�����ݼ�
            Instances newInstances = generateInstance(newData, classList, "sepalwidth", "petallength", "petalwidth", "class");
            newInstances.setClassIndex(newInstances.numAttributes() - 1);

            
            generateArffFile(newInstances, NEWIRIS_PATH + "new_iris.arff");

            System.out.println(newInstances);
            System.out.println("-------------------------------------");

//            RandomForest randomForest = new RandomForest(); //���ɭ�ַ�����
//            randomForest.buildClassifier(newInstances);
            
            Classifier classifier = new Logistic(); //�߼��ع������
            classifier.buildClassifier(newInstances); //ѵ��

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
	 *	��ȡ�ļ�    
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
