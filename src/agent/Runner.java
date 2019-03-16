package agent;

import javaclient.*;

public class Runner {
	/*
	 * Adapted from: https://github.com/Ryan-Amaral/working-gym-java-client
	 */
	public static void main(String[] args) {
		
		// Connect to the Gym Python server on the local host on port 5000.
		GymJavaHttpClient.baseUrl = "http://127.0.0.1:5000";
		
		// Create a new environment called CartPole-v0
		String id = GymJavaHttpClient.createEnv("CartPole-v0");

		// Reset the environment 
		Object obs = GymJavaHttpClient.resetEnv(id);

		System.out.println("obs:" +obs.toString());

		Environment e = new Environment();
		System.out.println("running experiments");
		e.runExperiments(id, obs);
		
		try {
			FileHandler.resultsToCSVFile(e.getTotalRewardEpisode(), id);
		} catch (Exception e1) {
			System.out.println("Unable to print to Excel.");
		}

	}
}
