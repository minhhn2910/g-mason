/**
 * Agent which have location as Double 2D must extend this
 * */ 

package sim.engine;
import sim.util.Double2D;
/** include index and set,get location method with Double2D
 * 
 * */

public abstract class AgentDouble
    {
		public Double2D loc;
		public int index;
		public int offset; //for multiple gpus
		public abstract Double2D getLocation();
		public abstract void setLocation(Double2D location);
    }
