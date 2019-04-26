package agent;

// Import the HTTP classes
import javaclient.*;

// Project Runner
public class Runner {
	
	// main function 
	public static void main(String[] args) {

		// Connect to the Gym Python server on the local host on port 5000.
		GymJavaHttpClient.baseUrl = "http://127.0.0.1:5000";

		// Create a new environment called CartPole-v0
		String id = GymJavaHttpClient.createEnv("CartPole-v0");

		// Reset the environment
		Object obs = GymJavaHttpClient.resetEnv(id);

		// Print initial observations in raw format
		System.out.println("obs:" + obs.toString());

		// Creating a new instance of Environment
		Environment env = new Environment();

		System.out.println("running experiments");
		// running the experiments
		env.runExperiments(id, obs);

		// send results to CSV
		FileHandler.resultsToCSVFile(env.getTotalRewardEpisode(), id);

	}
}
