package agent;

import javaclient.*;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;

public class Runner {
	/*
	 * Adapted from: https://github.com/Ryan-Amaral/working-gym-java-client
	 */
	public static void main(String[] args) {
		ArrayList<ArrayList<Float>> results = new ArrayList<ArrayList<Float>>();

		GymJavaHttpClient.baseUrl = "http://127.0.0.1:5000";

		String id = GymJavaHttpClient.createEnv("CartPole-v0");

		Object actionSpace = GymJavaHttpClient.actionSpace(id);

		System.out.println(actionSpace.getClass().getName());

		Object obs = GymJavaHttpClient.resetEnv(id);

		System.out.println(obs.getClass().getName());
		System.out.println(obs.toString());

		Environment e = new Environment();
		System.out.println("running experiments");
		e.runExperiments(id, obs);
		
		results.add(e.getTotalRewardEpisode());

		resultsToCSVFile(results, id);

	}

	private static void resultsToCSVFile(ArrayList<ArrayList<Float>> results, String id) {
		String resultsTable = resultsToCSVStr(results);
		try (PrintStream out = new PrintStream(
				new FileOutputStream("output/episodeReward.csv"))) {
			out.print(resultsTable);
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	// method to output experimental results in comma separated value format
	public static String resultsToCSVStr(ArrayList<ArrayList<Float>> results) {
		String output = "Episode No.,";
		for (int run = 0; run < results.size(); run++) {
			output += "Run" + run + "steps,";
		}
		for (int time = 0; time < results.get(0).size(); time++) {
			output += "\n" + time + ",";
			for (int run = 0; run < results.size(); run++) {
				output += "" + results.get(run).get(time) + ",";
			}
		}
		return output;
	}
}
