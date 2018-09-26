package aiLibsExperiments.baseExperiments;

import java.util.HashMap;
import java.util.Map;

import org.aeonbits.owner.ConfigCache;
import org.apache.commons.lang3.time.StopWatch;

import jaicore.basic.SQLAdapter;
import jaicore.experiments.ExperimentDBEntry;
import jaicore.experiments.IExperimentIntermediateResultProcessor;
import jaicore.experiments.IExperimentSetConfig;
import jaicore.experiments.IExperimentSetEvaluator;
import jaicore.ml.experiments.IMultiClassClassificationExperimentConfig;

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
		// Load dataset from openML
		
		// Split first test then val
		
		// Evaluate classifier on split
		StopWatch watch = new StopWatch();
		watch.start();
		double errorRate = 0;
		watch.stop();
		
		// Write results
		Map<String, Object> results = new HashMap<>();
		results.put("errorRate", errorRate);
		results.put("time", watch.getTime());
		results.put("val_seed", experimentEntry.getExperiment().getValuesOfKeyFields().get("test_seed"));		
		processor.processResults(results);
	}

}
