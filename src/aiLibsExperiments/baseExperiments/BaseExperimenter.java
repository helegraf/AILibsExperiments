package aiLibsExperiments.baseExperiments;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;

import org.aeonbits.owner.ConfigCache;
import org.apache.commons.lang3.time.StopWatch;
import org.openml.apiconnector.settings.Settings;

import de.upb.crc901.mlplan.multiclass.wekamlplan.weka.model.MLPipeline;
import jaicore.basic.SQLAdapter;
import jaicore.experiments.ExperimentDBEntry;
import jaicore.experiments.ExperimentRunner;
import jaicore.experiments.IExperimentIntermediateResultProcessor;
import jaicore.experiments.IExperimentSetConfig;
import jaicore.experiments.IExperimentSetEvaluator;
import jaicore.ml.WekaUtil;
import jaicore.ml.experiments.IMultiClassClassificationExperimentConfig;
import jaicore.ml.openml.OpenMLHelper;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.classifiers.AbstractClassifier;
import weka.core.Instances;

public class BaseExperimenter implements IExperimentSetEvaluator {

	// select for random selection
	// SELECT GROUP_CONCAT(dataset_id) AS random_selection FROM (SELECT * FROM
	// `dataset_set_members` WHERE dataset_set_name="small" AND
	// dataset_origin="openML_dataset_id" ORDER BY RAND() LIMIT 55) AS A GROUP BY
	// dataset_set_name

	// SELECT * FROM (SELECT * FROM `metafeature_runs` WHERE
	// dataset_origin="isys_id" )AS A INNER JOIN (SELECT * FROM metafeature_values
	// WHERE metafeature_name="NumberOfFeatures" OR
	// metafeature_name="NumberOfInstances") AS B ON
	// A.metafeature_run_id=B.metafeature_run_id

	// testset (isys-id: 36, 43, 34, 9, 3 / semeion, yeast, secom, car, abalone)

	IMultiClassClassificationExperimentConfig CONFIG = ConfigCache.getOrCreate(IBaseExperimenterConfig.class);

	@Override
	public IExperimentSetConfig getConfig() {
		return CONFIG;
	}

	@Override
	public void evaluate(ExperimentDBEntry experimentEntry, SQLAdapter adapter,
			IExperimentIntermediateResultProcessor processor) throws Exception {
		System.out.println("Read input values.");
		Map<String, String> valuesOfKeyFields = experimentEntry.getExperiment().getValuesOfKeyFields();
		BufferedReader reader = new BufferedReader(new InputStreamReader(
				Thread.currentThread().getContextClassLoader().getResourceAsStream("login_data")));
		reader.readLine();
		reader.readLine();
		reader.readLine();
		String openMLKey = reader.readLine();

		// Load dataset from openML
		System.out.println("Download dataset.");
		Settings.CACHE_ALLOWED = false;
		OpenMLHelper.setApiKey(openMLKey);
		Instances data = OpenMLHelper.getInstancesById(Integer.parseInt(valuesOfKeyFields.get("dataset_id")));

		// Evaluate classifier
		System.out.println("Evaluate Classifier");
		MLPipeline pipeline = new MLPipeline(ASSearch.forName(valuesOfKeyFields.get("searcher"), null),
				ASEvaluation.forName(valuesOfKeyFields.get("evaluator"), null),
				AbstractClassifier.forName(valuesOfKeyFields.get("classifier"), null));
		StopWatch watch = new StopWatch();
		watch.start();
		double errorRate = 1;
		try {
		errorRate = WekaUtil.evaluateClassifier(valuesOfKeyFields.get("split_technique"),
				valuesOfKeyFields.get("evaluation_technique"), Integer.parseInt(valuesOfKeyFields.get("seed")), data,
				pipeline);
		
		} catch (Exception e) {
			
		}
		watch.stop();
		
		// Write results
		System.out.println("Write results.");
		Map<String, Object> results = new HashMap<>();
		results.put("error_rate", errorRate);
		results.put("time", watch.getTime());
		processor.processResults(results);
		
		System.out.println("Done.");
	}

	public static void main(String[] args) {
		ExperimentRunner runner = new ExperimentRunner(new BaseExperimenter());
		runner.randomlyConductExperiments(false);
	}
}
