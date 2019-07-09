package aiLibsExperiments.small.miner;

import org.aeonbits.owner.Config.Sources;

import jaicore.ml.experiments.IMultiClassClassificationExperimentConfig;

@Sources({ "classpath:MetaMiner_small_Config.properties" })
public interface MetaMiner_small_Config extends IMultiClassClassificationExperimentConfig {
	
	public static final String DB_EVAL_TABLE = "db.evalTable";
	
	@Key(DB_EVAL_TABLE)
	@DefaultValue("intermediate_results_small_2")
	public String evaluationsTable();

}
