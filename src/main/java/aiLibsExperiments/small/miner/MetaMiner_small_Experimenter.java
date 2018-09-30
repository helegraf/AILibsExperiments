package aiLibsExperiments.small.miner;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.sql.ResultSet;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.aeonbits.owner.ConfigCache;
import org.apache.commons.io.IOUtils;

import com.google.common.eventbus.Subscribe;

import de.upb.crc901.mlplan.metamining.IntermediateSolutionEvent;
import de.upb.crc901.mlplan.metamining.MetaMLPlan;
import jaicore.basic.SQLAdapter;
import jaicore.experiments.ExperimentDBEntry;
import jaicore.experiments.ExperimentRunner;
import jaicore.experiments.IExperimentIntermediateResultProcessor;
import jaicore.experiments.IExperimentSetConfig;
import jaicore.experiments.IExperimentSetEvaluator;
import jaicore.ml.WekaUtil;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class MetaMiner_small_Experimenter implements IExperimentSetEvaluator {
	
	MetaMiner_small_Config CONFIG = ConfigCache.getOrCreate(MetaMiner_small_Config.class);
	long start;
	SQLAdapter adapter;
	double best = 1;
	int experimentID;
	
	public static void main(String[] args) {
		ExperimentRunner runner = new ExperimentRunner(new MetaMiner_small_Experimenter());
		runner.randomlyConductExperiments(false);
	}

	@Override
	public IExperimentSetConfig getConfig() {
		return CONFIG;
	}

	@Override
	public void evaluate(ExperimentDBEntry experimentEntry, SQLAdapter adapter,
			IExperimentIntermediateResultProcessor processor) throws Exception {
		/* load data for segment dataset and create a train-test-split */
		String query = "SELECT cluster_location_new, openML_dataset_id FROM dataset_id_mapping WHERE isys_id=?";
		List<String> values = Arrays.asList(experimentEntry.getExperiment().getValuesOfKeyFields().get("dataset_id"));
		ResultSet resultSet = adapter.getResultsOfQuery(query, values);
		resultSet.next();

		System.out.println("Queried for dataset information, isys_id: " + experimentEntry.getExperiment().getValuesOfKeyFields().get("dataset_id") + ".");
		//String cluster_location = resultSet.getString("cluster_location_new");
		// TODO only for testing!
		String cluster_location = "resources/credit-g_altered.arff";
//		switch(experimentEntry.getExperiment().getValuesOfKeyFields().get("dataset_id")) {
//		case "36" : cluster_location = "resources/semeion.arff"; break;
//		case "43" : cluster_location = "resources/yeast.arff";break;
//		case "34" : cluster_location = "resources/secom.arff";break;
//		case "9"  : cluster_location = "resources/car.arff";break;
//		case "3"  : cluster_location = "resources/abalone.arff";break;
//		}
		Instances data = new DataSource(cluster_location).getDataSet();
		data.setClassIndex(data.numAttributes()-1);
		List<Instances> split = WekaUtil.getStratifiedSplit(data, new Random(Integer.parseInt(experimentEntry.getExperiment().getValuesOfKeyFields().get("seed"))), .7f);
		this.experimentID = experimentEntry.getId();
		this.adapter = adapter;
		
		// Initialize meta mlplan and let it run for 2 minutes
		System.out.println("Example: Configure ML-Plan");
		File tempFile = File.createTempFile("tmp-metaminer-"+experimentEntry.getId(), "json");
		tempFile.deleteOnExit();
		FileOutputStream out = new FileOutputStream(tempFile);
		InputStream in = Thread.currentThread().getContextClassLoader().getResourceAsStream("weka-small.json");
		IOUtils.copy(in, out);
		File configFile = tempFile;
		MetaMLPlan metaMLPlan = new MetaMLPlan(configFile,data);
		metaMLPlan.setCPUs(CONFIG.getNumberOfCPUs());
		metaMLPlan.setTimeOutInMilliSeconds(Integer.parseInt(experimentEntry.getExperiment().getValuesOfKeyFields().get("timeout"))*1000);
		metaMLPlan.setSeed(Integer.parseInt(experimentEntry.getExperiment().getValuesOfKeyFields().get("seed")));
		metaMLPlan.setMetaFeatureSetName("all");
		metaMLPlan.setDatasetSetName("metaminer_small");
		metaMLPlan.registerListenerForIntermediateSolutions(this);
		
		// Build meta components
		System.out.println("Example: build meta components");
		metaMLPlan.buildMetaComponents(CONFIG.getDBHost(), CONFIG.getDBUsername(), CONFIG.getDBPassword());

		// Build classifier
		System.out.println("Example: find solution");
		start = System.currentTimeMillis();
		metaMLPlan.buildClassifier(split.get(0));
		long trainTime = System.currentTimeMillis() - start;

		// Evaluate solution produced by meta mlplan
		System.out.println("Example: ");
		Evaluation eval = new Evaluation(split.get(0));
		eval.evaluateModel(metaMLPlan, split.get(1));
		System.out.println("Error Rate of the solution produced by Meta ML-Plan: " + (100 - eval.pctCorrect()) / 100f);
		
		Map<String, Object> results = new HashMap<>();
		results.put("error_rate", (100 - eval.pctCorrect())/100f);
		results.put("time", trainTime);
		processor.processResults(results);		
	}
	
	@Subscribe
	public void uploadIntermediateSolution(IntermediateSolutionEvent e) {
		if (this.adapter != null && e.getScore() < best) {
			try {
				best = e.getScore();
				Map<String, Object> eval = new HashMap<>();
				eval.put("run_id", this.experimentID);
				eval.put("algorithm", "Meta-Miner");
				if (e.getSearcher() != null && e.getEvaluator() != null) {
					eval.put("searcher", e.getSearcher());
					eval.put("evaluator", e.getEvaluator());
				}
				eval.put("classifier", e.getClassifier());
				eval.put("found_at", (int)(System.currentTimeMillis() - start));
				eval.put("score", e.getScore());

				this.adapter.insert_noAutoGeneratedFields(CONFIG.evaluationsTable(), eval);
			} catch (Exception e1) {
				e1.printStackTrace();
			}
		}
	}

}
