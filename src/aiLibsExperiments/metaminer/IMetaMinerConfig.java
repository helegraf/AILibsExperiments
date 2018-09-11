package aiLibsExperiments.metaminer;
import org.aeonbits.owner.Config.Sources;

import jaicore.ml.experiments.IMultiClassClassificationExperimentConfig;

@Sources({ "file:./conf/setup.properties" })
public interface IMetaMinerConfig extends IMultiClassClassificationExperimentConfig {
	
}
