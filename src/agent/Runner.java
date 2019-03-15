package agent;

import javaclient.*;

public class Runner {
	/*
	 * Adapted from: https://github.com/Ryan-Amaral/working-gym-java-client
	 */
	public static void main(String[] args) {

		GymJavaHttpClient.baseUrl = "http://127.0.0.1:5000";

		String id = GymJavaHttpClient.createEnv("CartPole-v0");

		Object obs = GymJavaHttpClient.resetEnv(id);

		System.out.println(obs.getClass().getName());
		System.out.println(obs.toString());

		Environment e = new Environment();
		System.out.println("running experiments");
		e.runExperiments(id, obs);
		
		FileHandler.resultsToCSVFile(e.getTotalRewardEpisode(), id);

	}
}
