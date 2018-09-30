package aiLibsExperiments.metaminer;

import java.util.HashMap;

import org.aeonbits.owner.ConfigCache;
import org.apache.commons.lang.time.StopWatch;

import de.upb.crc901.mlplan.metamining.MetaMLPlan;
import jaicore.basic.SQLAdapter;
import jaicore.experiments.Experiment;
import jaicore.experiments.ExperimentDBEntry;
import jaicore.experiments.ExperimentRunner;
import jaicore.experiments.IExperimentIntermediateResultProcessor;
import jaicore.experiments.IExperimentSetConfig;
import jaicore.experiments.IExperimentSetEvaluator;
import jaicore.ml.WekaUtil;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class MetaMinerExperimenter implements IExperimentSetEvaluator {

	private IMetaMinerConfig CONFIG = ConfigCache.getOrCreate(IMetaMinerConfig.class);

	@Override
	public void evaluate(ExperimentDBEntry experimentEntry, SQLAdapter adapter,
			IExperimentIntermediateResultProcessor processor) throws Exception {

		//TODO pass/ integrate split technique and seed
		Experiment experiment = experimentEntry.getExperiment();
		String split_technique = experiment.getValuesOfKeyFields().get("split_technique");
		int seed = Integer.parseInt(experiment.getValuesOfKeyFields().get("seed"));

		// Get data set and split 
		System.out.println("Experimenter: Starting Experiment with id: " + experimentEntry.getId());
		Instances data = new DataSource(experiment.getValuesOfKeyFields().get("dataset_id")).getDataSet();
		Instances train = WekaUtil.getTrainSplit(split_technique, data, seed);
		Instances test = WekaUtil.getTestSplit(split_technique, data, seed);

		// Set up metaminer
		MetaMLPlan metaminer = new MetaMLPlan(train);
		metaminer.setMetaFeatureSetName(experiment.getValuesOfKeyFields().get("metafeature_set"));
		metaminer.setDatasetSetName(experiment.getValuesOfKeyFields().get("dataset_set"));
		metaminer.setTimeOutInMilliSeconds(
				Integer.parseInt(experiment.getValuesOfKeyFields().get("timeout_in_seconds")) * 1000);
		metaminer.setCPUs(experiment.getNumCPUs());

		// Build meta components
		System.out.println("Experimenter: Building Meta Components.");
		StopWatch watch = new StopWatch();
		HashMap<String, Object> results = new HashMap<>(3);
		watch.start();
		metaminer.buildMetaComponents(CONFIG.getDBHost(), CONFIG.getDBUsername(), CONFIG.getDBPassword());
		watch.stop();
		results.put("meta_build_time", watch.getTime());
		watch.reset();

		// Search for pipeline
		System.out.println("Building Classifier (starting search).");
		watch.start();
		metaminer.buildClassifier(train);
		watch.stop();
		results.put("built_time", watch.getTime());
		watch.reset();

		// Evaluate final pipeline
		System.out.println("Experimenter: Evaluating Classifier.");
		Evaluation eval = new Evaluation(data);
		eval.evaluateModel(metaminer, test);

		// Process results
		System.out.println("Experimenter: Classifier has error rate " + (1 - eval.pctCorrect()) + ".");
		results.put("loss", 1 - eval.pctCorrect());
		processor.processResults(results);
	}

	@Override
	public IExperimentSetConfig getConfig() {
		return this.CONFIG;
	}

	public static void main(String[] args) {
		ExperimentRunner runner = new ExperimentRunner(new MetaMinerExperimenter());
		runner.randomlyConductExperiments(false);
	}
}
