package aiLibsExperiments.metafeatures;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.sql.ResultSet;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.openml.apiconnector.settings.Settings;

import jaicore.basic.SQLAdapter;
import jaicore.ml.metafeatures.GlobalCharacterizer;
import jaicore.ml.openml.OpenMLHelper;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Main {

	public static void main(String[] args) throws Exception {
		//TODO remember to copy the current build to the cluster and let it run for the last dataset!
	}
	
	public static void adaptMetafeatures(String [] args) throws Exception {
		int offset = Integer.parseInt(args[0]);

		BufferedReader reader = new BufferedReader(new InputStreamReader(
				Thread.currentThread().getContextClassLoader().getResourceAsStream("login_data")));
		String host = reader.readLine();
		String user = reader.readLine();
		String password = reader.readLine();
		String openMLKey = reader.readLine();
		reader.close();

		SQLAdapter adapter_metafeatures = new SQLAdapter(host, user, password, "hgraf");
		SQLAdapter adapter_jobs = new SQLAdapter(host, user, password, "pgotfml_hgraf");

		// Get cluster location and openML id
		String query = "SELECT cluster_location_new, openML_dataset_id FROM dataset_id_mapping WHERE isys_id=?";
		List<String> values = Arrays.asList(String.valueOf(offset));
		ResultSet resultSet = adapter_jobs.getResultsOfQuery(query, values);
		resultSet.next();

		int dataset_id = offset;

		System.out.println("Queried for dataset information, isys_id: " + dataset_id + ".");

		String dataset_origin = "isys_id";
		String cluster_location = resultSet.getString("cluster_location_new");
		// TODO only for testing!
		// String cluster_location = "resources/credit-g_altered.arff";
		Instances data = new DataSource(cluster_location).getDataSet();
		if (data == null) {
			throw new FileNotFoundException("Could not find dataset at " + cluster_location);
		}
		data.setClassIndex(data.numAttributes() - 1);

		System.out.println("Loaded dataset from cluster local file.");

		if (resultSet.getObject("openML_dataset_id") != null) {
			int openMLId = resultSet.getInt("openML_dataset_id");

			System.out.println("Dataset has openML id: " + openMLId);

			// Check if cluster dataset equal to openML dataset
			Instances cluster_dataset = data;

			Settings.CACHE_ALLOWED = false;
			OpenMLHelper.setApiKey(openMLKey);
			try {
			Instances openML_dataset = OpenMLHelper.getInstancesById(resultSet.getInt("openML_dataset_id"));
			
			System.out.println("Got dataset from openML. Checking for equality.");

			if (cluster_dataset.numAttributes() == openML_dataset.numAttributes()
					&& cluster_dataset.numInstances() == openML_dataset.numInstances()) {
				boolean equal = true;
				isEqual: for (int i = 0; i < cluster_dataset.numInstances(); i++) {
					for (int j = 0; j < cluster_dataset.numAttributes(); j++) {

						if (cluster_dataset.get(i).value(j) != openML_dataset.get(i).value(j)) {
							System.out.println("Datasets not equal! Instance " + i + " Attribute " + j
									+ " unequal. Cluster: " + cluster_dataset.get(i).value(j) + " openML: "
									+ openML_dataset.get(i).value(j));
							adapter_jobs.update(
									"UPDATE `dataset_id_mapping` SET `openML_dataset_id` = NULL WHERE `dataset_id_mapping`.`isys_id` = "
											+ String.valueOf(offset));
							equal = false;
							break isEqual;
						}
					}
				}

				if (equal) {
					dataset_id = openMLId;
					dataset_origin = "openML_dataset_id";
					System.out.println("Datasets are equal! Creating entry in metafeatures db under openML key.");
				}

			} else {
				adapter_jobs.update(
						"UPDATE `dataset_id_mapping` SET `openML_dataset_id` = NULL WHERE `dataset_id_mapping`.`isys_id` = "
								+ String.valueOf(offset));
				System.out.println("Datasets not equal! Unequal number of instances / attributes.");
			}
			} catch (IOException e) {
				System.out.println("Failure while reading openML instances object. Considering dataset unequal");
				adapter_jobs.update(
						"UPDATE `dataset_id_mapping` SET `openML_dataset_id` = NULL WHERE `dataset_id_mapping`.`isys_id` = "
								+ String.valueOf(offset));
			}

		}
		resultSet.close();

		// Check if there are already results for the dataset
		boolean createdJobExists = false;
		int createdJobId = 0;

		System.out.println("Check for existing evaluation data.");
		query = "SELECT metafeature_run_id, status FROM metafeature_runs WHERE dataset_id=? AND dataset_origin=?";
		values = Arrays.asList(String.valueOf(dataset_id), dataset_origin);
		resultSet = adapter_metafeatures.getResultsOfQuery(query, values);

		if (resultSet.next()) {
			if (resultSet.getString("status").equals("finished")) {
				System.out.println("Evaluation data exists! Job finished.");
				return;
			} else {
				createdJobExists = true;
				createdJobId = resultSet.getInt("metafeature_run_id");
				System.out.println("Resuming existing evaluation. Id: " + createdJobId);
			}
		}
		resultSet.close();

		// Check if there exists an entry for the data set in the ids! If not, add it.
		System.out.println("Check for dataset_id entry for id " + dataset_id + " origin: " + dataset_origin);
		query = "SELECT * FROM dataset_ids WHERE dataset_id=? AND dataset_origin=?";
		values = Arrays.asList(String.valueOf(dataset_id), dataset_origin);
		resultSet = adapter_metafeatures.getResultsOfQuery(query, values);

		if (resultSet.next()) {
			// Entry exists
			System.out.println("Entry exists.");
		} else {
			// Entry does not exist
			System.out.println("Entry does not exists. Creating entry.");
			HashMap<String, Object> map = new HashMap<>();
			map.put("dataset_id", dataset_id);
			map.put("dataset_origin", dataset_origin);
			map.put("dataset_name", data.relationName().length() <= 300 ? data.relationName()
					: data.relationName().substring(0, 297) + "...");
			adapter_metafeatures.insert_noAutoGeneratedFields("dataset_ids", map);
			System.out.println("Entry created.");
		}
		resultSet.close();

		// Compute meta data and put online
		System.out.println("Characterize data.");
		GlobalCharacterizer chara = new GlobalCharacterizer();
		Map<String, Double> characterization = chara.characterize(data);

		// Create job is not exists
		if (!createdJobExists) {
			System.out.println("Creating new job entry.");
			HashMap<String, String> map = new HashMap<>();
			map.put("dataset_id", String.valueOf(dataset_id));
			map.put("dataset_origin", dataset_origin);
			createdJobId = adapter_metafeatures.insert("metafeature_runs", map);
			System.out.println("Created new job for evaluation data. Id: " + createdJobId);
		}

		// Update values
		System.out.println("Updating metafeature values.");
		for (String metafeature : characterization.keySet()) {
			HashMap<String, Object> insertValues = new HashMap<>();
			insertValues.put("metafeature_run_id", String.valueOf(createdJobId));
			insertValues.put("metafeature_name", metafeature);
			insertValues.put("metafeature_value", characterization.get(metafeature));
			System.out.println("Adding metafeature: " + metafeature + " value: " + characterization.get(metafeature));
			adapter_metafeatures.insert_noAutoGeneratedFields("metafeature_values", insertValues);
		}

		// Update times
		System.out.println("Updating metafeature times.");
		for (String metafeatureGroup : chara.getMetaFeatureComputationTimes().keySet()) {
			HashMap<String, Object> insertValues = new HashMap<>();
			insertValues.put("metafeature_run_id", createdJobId);
			insertValues.put("metafeature_group", metafeatureGroup.replace(", 2 folds", ""));
			insertValues.put("time", chara.getMetaFeatureComputationTimes().get(metafeatureGroup));
			System.out.println("Adding time: " + metafeatureGroup.replace(", 2 folds", "") + " value: "
					+ chara.getMetaFeatureComputationTimes().get(metafeatureGroup));
			adapter_metafeatures.insert_noAutoGeneratedFields("metafeature_times", insertValues);
		}

		// Update job status
		System.out.println("Updating job status.");
		HashMap<String, Object> updateValues = new HashMap<>();
		updateValues.put("status", "finished");
		HashMap<String, Object> conditions = new HashMap<>();
		conditions.put("metafeature_run_id", String.valueOf(createdJobId));
		adapter_metafeatures.update("metafeature_runs", updateValues, conditions);
		System.out.println("Finished.");
	}
}
