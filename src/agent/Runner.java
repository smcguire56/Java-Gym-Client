package agent;

import javaclient.*;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.json.JSONArray;
import org.json.JSONObject;

public class Runner {
	/*
	 * Adapted from: https://github.com/Ryan-Amaral/working-gym-java-client
	 */
	public static void main(String[] args) {
		System.out.println("running main");

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
	}
}
