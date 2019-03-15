package agent;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;

public class FileHandler {
	
	public FileHandler() {
		super();
	}

	static void resultsToCSVFile(ArrayList<ArrayList<Float>> results, String id) {
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
