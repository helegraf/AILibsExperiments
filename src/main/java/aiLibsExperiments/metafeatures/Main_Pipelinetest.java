package aiLibsExperiments.metafeatures;

import de.upb.crc901.mlplan.multiclass.wekamlplan.weka.model.MLPipeline;
import weka.attributeSelection.GainRatioAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Main_Pipelinetest {
	public static void main(String[] args) throws Exception {
		// TODO remove
		MultilayerPerceptron perceptron = new MultilayerPerceptron();
		perceptron.setLearningRate(0.7);
		MLPipeline pipeline = new MLPipeline(new Ranker(), new GainRatioAttributeEval(), perceptron);
		Instances data = new DataSource("resources/credit-g_altered.arff").getDataSet();

		pipeline.buildClassifier(data);
		pipeline.classifyInstance(data.get(0));

//		MLPipelineComponentInstanceFactory factory = new MLPipelineComponentInstanceFactory(
//				new File("resources/weka-all-autoweka.json"));
//		
//		//ComponentInstance CI = factory.convertToComponentInstance(new MLPipeline(
//		//		"[SupervisedFilterSelector [searcher=weka.attributeSelection.Ranker, evaluator=weka.attributeSelection.ReliefFAttributeEval]] (preprocessors), weka.classifiers.functions.Logistic- [-R, 1.0E-9, -M, -1, -num-decimal-places, 4] (classifier)"));
//
//		WEKAPipelineCharacterizer chara = new WEKAPipelineCharacterizer(factory.getLoader().getParamConfigs());
//		chara.characterize(factory.convertToComponentInstance(pipeline));
	}
}
