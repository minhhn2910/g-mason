	/*
  Copyright 2006 by Sean Luke and George Mason University
  Licensed under the Academic Free License version 3.0
  See the file "LICENSE" for more information
*/

package sim.app.flockers;


//import for jcuda
//
//
import static jcuda.driver.JCudaDriver.*;
import static jcuda.runtime.JCuda.*;

import java.util.Arrays;
import jcuda.jcurand.*;
import jcuda.runtime.JCuda;

import java.io.*;
import jcuda.*;
import jcuda.driver.*;

//end import for jcuda
import java.io.*;
import java.nio.ByteOrder;
import java.nio.ByteBuffer; //byte buffer for better copy time, but the code will be more complicated
import java.nio.DoubleBuffer;

import sim.engine.*;
import sim.util.*;
import sim.field.continuous.*;




public class Flockers extends SimState
    {
    private static final long serialVersionUID = 1;

    public Continuous2D flockers;
    
    public double width = 8000;
    public double height = 8000;

    public int numFlockers = 2500000;
    public double cohesion = 1.0;
    public double avoidance = 1.0;
    public double randomness = 1.0;
    public double consistency = 1.0;
    public double momentum = 1.0;
    public double deadFlockerProbability = 0.1;
    public double neighborhood = 10;
    public double jump = 0.7;  // how far do we move in a timestep?

    
    public int numDeads = 0;
    public int numActiveFlockers;

	public int is_dead[];//constants, need to copy to device once


	//for better performance, combine all double vector to 1
	
	//Number of devices here
	private final int numDevices =2;
	
	
	public Pointer[] dataPointer;
	public DoubleBuffer[] data;
	// = new double[numDevices][(numFlockers*4)/numDevices]; //that is every flocker has 4 elements for data, divide by 2 => 2 array, 4*(numElement/2)
	//follow this sequence : agent1(loc_x, loc_y, dx, dy), agent2 .... for coalescing the GPU memory, better performance
	
	public FlockersGPUContext Devices[];


      
    public double getCohesion() { return cohesion; }
    public void setCohesion(double val) { if (val >= 0.0) cohesion = val; }
    public double getAvoidance() { return avoidance; }
    public void setAvoidance(double val) { if (val >= 0.0) avoidance = val; }
    public double getRandomness() { return randomness; }
    public void setRandomness(double val) { if (val >= 0.0) randomness = val; }
    public double getConsistency() { return consistency; }
    public void setConsistency(double val) { if (val >= 0.0) consistency = val; }
    public double getMomentum() { return momentum; }
    public void setMomentum(double val) { if (val >= 0.0) momentum = val; }
    public int getNumFlockers() { return numFlockers; }
    public void setNumFlockers(int val) { if (val >= 1) numFlockers = val; }
    public double getWidth() { return width; }
    public void setWidth(double val) { if (val > 0) width = val; }
    public double getHeight() { return height; }
    public void setHeight(double val) { if (val > 0) height = val; }
    public double getNeighborhood() { return neighborhood; }
    public void setNeighborhood(double val) { if (val > 0) neighborhood = val; }
    public double getDeadFlockerProbability() { return deadFlockerProbability; }
    public void setDeadFlockerProbability(double val) { if (val >= 0.0 && val <= 1.0) deadFlockerProbability = val; }
    

    public Double2D[] getLocations()
        {
        if (flockers == null) return new Double2D[0];
        Bag b = flockers.getAllObjects();
        if (b==null) return new Double2D[0];
        Double2D[] locs = new Double2D[b.numObjs];
        for(int i =0; i < b.numObjs; i++)
            locs[i] = flockers.getObjectLocation(b.objs[i]);
        return locs;
        }
    
    public Double2D[] getInvertedLocations()
        {
        if (flockers == null) return new Double2D[0];
        Bag b = flockers.getAllObjects();
        if (b==null) return new Double2D[0];
        Double2D[] locs = new Double2D[b.numObjs];
        for(int i =0; i < b.numObjs; i++)
            {
            locs[i] = flockers.getObjectLocation(b.objs[i]);
            locs[i] = new Double2D(locs[i].y, locs[i].x);
            }
        return locs;
        }

    /** Creates a Flockers simulation with the given random number seed. */
    public Flockers(long seed)
        {
        super(seed);
        }
    
    public void start()
        {
        super.start();
        
        System.out.println("here");
        
        is_dead  = new int[numFlockers]; //constants, required copy to device once
        long memorySize = (numFlockers*4)/numDevices*Sizeof.DOUBLE;
        
        double temp_fill_data[] = new double[(numFlockers*4)/numDevices];
        Arrays.fill(temp_fill_data, 0);
        
        dataPointer = new Pointer[numDevices];
        data = new DoubleBuffer[numDevices];
        
        
        for(int i =0; i<numDevices;i++)
        {
			dataPointer[i] = new Pointer();
			cudaHostAlloc(dataPointer[i], memorySize, cudaHostAllocPortable); //multiple devices, multiple contexts
			data[i] = dataPointer[i].getByteBuffer(0, memorySize).order(ByteOrder.nativeOrder()).asDoubleBuffer();
			data[i].put(temp_fill_data);
        }
        
        int numElements = numFlockers/numDevices;
        
        
        // set up the flockers field.  It looks like a discretization
        // of about neighborhood / 1.5 is close to optimal for us.  Hmph,
        // that's 16 hash lookups! I would have guessed that 
        // neighborhood * 2 (which is about 4 lookups on average)random
        // would be optimal.  Go figure.
        flockers = new Continuous2D(neighborhood/1.5,width,height);
        numDeads = 0;
        // make a bunch of flockers and schedule 'em.  A few will be dead		
		
        for(int x=0;x<numFlockers;x++)
            {
				
            Double2D location = new Double2D(random.nextDouble()*width, random.nextDouble() * height);
			Flocker flocker = new Flocker(location);
			
            is_dead[x] = 0;
            if (random.nextBoolean(deadFlockerProbability))
            {
				 flocker.dead = true;
				 is_dead[x] = 1;
				 numDeads++;
			}
			
            flockers.initObjectLocation(flocker, location);
            flocker.flockers = flockers;
            flocker.theFlock = this;

			flocker.offset = x/numElements;
			flocker.index = (x - flocker.offset*numFlockers/numDevices)*4;
						
            flocker.setLocation(location);  
			//just need to schedule one anonymous agent to keep Mason schedule "heart" still beating in discrete time
     
            }
  
           
      	schedule.scheduleRepeating(schedule.EPOCH, 1, new Steppable()
		{
			public void step(SimState state) { return; }
		});
		
		      
        numActiveFlockers = numFlockers - numDeads;
		schedule.currentPhase = 2;
		System.out.println("flockers " + numFlockers + "  dead " +numDeads) ;

		//arrange your GPU stuff here, inside a async thread 
	
        AsynchronousSteppable s = new AsynchronousSteppable()
		{
			boolean shouldQuit = false;
			int gpu_step = 0;
			int schedule_gpuSteps =0;
			Object[] lock = new Object[0]; // an array is a unique, serializable object
			protected void run(boolean resuming)
			{
				boolean quit = false;
				if (!resuming)
				{	
					cuda_init();
					schedule_gpuSteps = schedule.getgpuSteps();
					
					System.out.println("Async agent started");
					// we’re starting fresh -- set up here if you have to
			
				}
				else // (resuming)
				{
					//carefully check here

					cuda_init();
					schedule_gpuSteps = schedule.getgpuSteps();
					
					System.out.println("Async agent resumed");
					// we’re starting fresh -- set up here if you have to
			//		my_start_time = System.currentTimeMillis();
				}
				while(!quit)
				{
					if(schedule.currentPhase == 2)  //GPU phases
					{

					
						for(int i=0; i<numDevices;i++)
						{
							Devices[i].eventSync();
							Devices[i].launchKernel();
							Devices[i].eventRecord();
						}
						//		System.out.println("setup success" );
						//copy back
						for(int i=0; i<numDevices;i++)
							Devices[i].eventSync();

						gpu_step ++;
						if(gpu_step == schedule_gpuSteps) //time to copy back and wake up mason's Schedule
						{
					
						
							for(int i=0; i<numDevices;i++)
								Devices[i].copy_back(dataPointer);						
								gpu_step = 0;
								schedule.return_to_CPU();
						 }
			
			
						for(int i=0; i<numDevices;i++)
						{
							Devices[i].sync_data_peer(Devices);
							Devices[i].eventRecord();
						}
						
						for(int i=0; i<numDevices;i++)
						{
							Devices[i].eventSync();
							Devices[i].setup_continuous2D();
							Devices[i].sync_internal_data();
							Devices[i].eventRecord();
						}
	
					}
						synchronized(lock) { quit = shouldQuit; shouldQuit = false; }
					}
				//quit here
				System.out.println("Free cuda mem");
				cudaDeviceReset();
				System.out.println("Cleanup success");
				
				}
				protected void halt(boolean pausing) { synchronized(lock) { shouldQuit = true; } }
			};
			
			
			s.step(this);   
            
            
            
        }

    public static void main(String[] args)
        {
        doLoop(Flockers.class, args);
        System.exit(0);
        }  
     
    public void cuda_init() 
        {

                JCudaDriver.setExceptionsEnabled(true);
                JCuda.setExceptionsEnabled(true);
                JCurand.setExceptionsEnabled(true);
				cudaDeviceReset();
				cuInit(0);
				Devices = new FlockersGPUContext[numDevices];  //array of GPU contexts, can be applied to multiple GPUs
				int mod = numFlockers % numDevices; //solve numAgents not devided by numDevices
				int numElements = numFlockers/numDevices;
				int lastNumElements = 0;
				for(int i = 0; i<numDevices; i++)
				{
					Devices[i] = new FlockersGPUContext(i,i);//only device 0 for test in 1 GPU machine, index for simulate that it has 4 devices
					Devices[i].InitFlockersConstant(cohesion, avoidance, randomness, consistency, momentum, jump);
					if(mod !=0)
					{ 
						lastNumElements=(numElements+1);
						Devices[i].InitModule(1234/*can put schedule.seed here*/, numElements+1, numFlockers, width, height, neighborhood, numDevices); 
						mod=mod-1;
					}
					else
					{
						lastNumElements=numElements;			
						Devices[i].InitModule(1234/*can put schedule.seed here*/, numElements, numFlockers, width, height, neighborhood,numDevices); 
					}
					
					Devices[i].InitMemory(dataPointer, is_dead);
					Devices[i].InitKernelParameters();
					
					System.out.println("device " +Devices[i].index + " numelements "+Devices[i].numElements + " offset " +Devices[i].offset + " bytes " + Devices[i].bytecount);
				}
				System.out.println("initial CUDA success");
		
	//	pre setup field 
						for(int i=0; i<numDevices;i++)
						{
							Devices[i].setup_continuous2D();
							Devices[i].eventRecord();
						}
		
		
		}
		
		
	

}
