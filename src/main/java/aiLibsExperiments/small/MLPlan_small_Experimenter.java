package aiLibsExperiments.small;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.sql.ResultSet;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Random;

import org.aeonbits.owner.ConfigCache;
import org.apache.commons.io.IOUtils;

import com.google.common.eventbus.Subscribe;

import de.upb.crc901.mlplan.multiclass.wekamlplan.MLPlanWekaBuilder;
import de.upb.crc901.mlplan.multiclass.wekamlplan.MLPlanWekaClassifier;
import de.upb.crc901.mlplan.multiclass.wekamlplan.weka.WEKAPipelineFactory;
import de.upb.crc901.mlplan.multiclass.wekamlplan.weka.WekaMLPlanWekaClassifier;
import de.upb.crc901.mlplan.multiclass.wekamlplan.weka.model.MLPipeline;
import hasco.core.HASCOSolutionCandidate;
import jaicore.basic.SQLAdapter;
import jaicore.basic.algorithm.SolutionCandidateFoundEvent;
import jaicore.experiments.ExperimentDBEntry;
import jaicore.experiments.ExperimentRunner;
import jaicore.experiments.IExperimentIntermediateResultProcessor;
import jaicore.experiments.IExperimentSetConfig;
import jaicore.experiments.IExperimentSetEvaluator;
import jaicore.ml.WekaUtil;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class MLPlan_small_Experimenter implements IExperimentSetEvaluator {

	MLPlan_small_Config CONFIG = ConfigCache.getOrCreate(MLPlan_small_Config.class);
	WEKAPipelineFactory factory = new WEKAPipelineFactory();
	int experimentID;
	SQLAdapter adapter;
	long start;
	double best = 1;
	
	public static void main(String[] args) {
		ExperimentRunner runner = new ExperimentRunner(new MLPlan_small_Experimenter());
		runner.randomlyConductExperiments(false);
	}

	@Override
	public IExperimentSetConfig getConfig() {
		return this.CONFIG;
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
//		case "36" : cluster_location = "resources/semeion.arff";break;
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

		/* initialize mlplan, and let it run for 30 seconds */
		MLPlanWekaBuilder standard = new MLPlanWekaBuilder();
		File tempFile = File.createTempFile("tmp-metaminer-"+experimentEntry.getId(), "json");
		tempFile.deleteOnExit();
		FileOutputStream out = new FileOutputStream(tempFile);
		InputStream in = Thread.currentThread().getContextClassLoader().getResourceAsStream("weka-small.json");
		IOUtils.copy(in, out);
		File configFile = tempFile;
		MLPlanWekaBuilder builder = new MLPlanWekaBuilder(
				configFile, standard.getAlhorithmConfigFile(),
				standard.getPerformanceMeasure());
		builder.setSeed(Integer.parseInt(experimentEntry.getExperiment().getValuesOfKeyFields().get("seed")));
		MLPlanWekaClassifier mlplan = new WekaMLPlanWekaClassifier(builder);
		mlplan.setLoggerName("mlplan");
		mlplan.setTimeout(Integer.parseInt(experimentEntry.getExperiment().getValuesOfKeyFields().get("timeout")));
		mlplan.registerListenerForSolutionEvaluations(this);
		try {
			start = System.currentTimeMillis();
			mlplan.buildClassifier(split.get(0));
			long trainTime = (int) (System.currentTimeMillis() - start) / 1000;
			System.out.println("Finished build of the classifier. Training time was " + trainTime + "s.");

			/* evaluate solution produced by mlplan */
			Evaluation eval = new Evaluation(split.get(0));
			eval.evaluateModel(mlplan, split.get(1));
			System.out.println("Error Rate of the solution produced by ML-Plan: " + (100 - eval.pctCorrect())/100f);
			
			Map<String, Object> results = new HashMap<>();
			results.put("error_rate", (100 - eval.pctCorrect())/100f);
			results.put("time", trainTime);
			processor.processResults(results);
		} catch (NoSuchElementException e) {
			System.out.println("Building the classifier failed: " + e.getMessage());
		}

	}

	@Subscribe
	public void rcvHASCOSolutionEvent(final SolutionCandidateFoundEvent<HASCOSolutionCandidate<Double>> e) {
		if (this.adapter != null && e.getSolutionCandidate().getScore() < best) {
			try {
				best = e.getSolutionCandidate().getScore();
				MLPipeline pl = this.factory.getComponentInstantiation(e.getSolutionCandidate().getComponentInstance());
				Map<String, Object> eval = new HashMap<>();
				eval.put("run_id", this.experimentID);
				eval.put("algorithm", "ML-Plan");
				if (pl.getPreprocessors() != null && pl.getPreprocessors().size() > 0) {
					eval.put("searcher", pl.getPreprocessors().get(0).getSearcher().getClass().getName());
					eval.put("evaluator", pl.getPreprocessors().get(0).getEvaluator().getClass().getName());
				}
				eval.put("classifier",pl.getBaseClassifier().getClass().getName());
				eval.put("found_at", (int)(System.currentTimeMillis() - start));
				eval.put("score", e.getSolutionCandidate().getScore());

				this.adapter.insert_noAutoGeneratedFields(CONFIG.evaluationsTable(), eval);
			} catch (Exception e1) {
				e1.printStackTrace();
			}
		}
	}

}
