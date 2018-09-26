package aiLibsExperiments.baseExperiments;

import org.aeonbits.owner.Config.Sources;

import jaicore.ml.experiments.IMultiClassClassificationExperimentConfig;

@Sources({ "file:./conf/baseExperimenterConfig.properties" })
public interface IBaseExperimenterConfig extends IMultiClassClassificationExperimentConfig {

}
