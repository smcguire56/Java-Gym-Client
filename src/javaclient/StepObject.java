package javaclient;

public class StepObject {
	// Local variables for the step object
	public Object observation;
	public float reward;
	public boolean done;
	public Object info;

	// Default constructor for step object, allowing reading in from server and
	// converting to step object
	public StepObject(Object observation, float reward, boolean done, Object info) {
		this.observation = observation;
		this.reward = reward;
		this.done = done;
		this.info = info;
	}
}
