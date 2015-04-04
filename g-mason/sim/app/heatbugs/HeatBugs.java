/*
  Copyright 2006 by Sean Luke and George Mason University
  Licensed under the Academic Free License version 3.0
  See the file "LICENSE" for more information
*/
/*
 * HeatBugs single GPU version for simpler code
 * the MultiGPU version can be found in sim/app/heatbugsMultiGPU/
 * */
package sim.app.heatbugs;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.runtime.JCuda.*;
import java.util.Arrays;
import jcuda.jcurand.*;
import jcuda.runtime.JCuda;
import java.io.*;
import jcuda.*;
import jcuda.driver.*;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;

import sim.engine.*;
import sim.field.grid.*;
import sim.util.*;

public /*strictfp*/ class HeatBugs extends SimState
    {
    private static final long serialVersionUID = 1;

    public double minIdealTemp = 17000;
    public double maxIdealTemp = 31000;
    public double minOutputHeat = 6000;
    public double maxOutputHeat = 10000;

    public double evaporationRate = 0.993;
    public double diffusionRate = 1.0;
    public static final double MAX_HEAT = 32000;
    public double randomMovementProbability = 0.1;

    public int gridHeight;
    public int gridWidth;
    public int bugCount;
    
    HeatBug[] bugs;
 
	public HeatBugsGPUContext context;

	//public int [] loc_x ;//= new int[bugCount];
	//public int [] loc_y ;//= new int[bugCount];
	public Pointer locationPointer;
	public IntBuffer location;
	
	
	public double[] idealTemp;// = new double[bugCount];
    public double[] heatOutput;// = new double[bugCount];
    
    
    public double getMinimumIdealTemperature() { return minIdealTemp; }
    public void setMinimumIdealTemperature( double temp ) { if( temp <= maxIdealTemp ) minIdealTemp = temp; }
    public double getMaximumIdealTemperature() { return maxIdealTemp; }
    public void setMaximumIdealTemperature( double temp ) { if( temp >= minIdealTemp ) maxIdealTemp = temp; }
    public double getMinimumOutputHeat() { return minOutputHeat; }
    public void setMinimumOutputHeat( double temp ) { if( temp <= maxOutputHeat ) minOutputHeat = temp; }
    public double getMaximumOutputHeat() { return maxOutputHeat; }
    public void setMaximumOutputHeat( double temp ) { if( temp >= minOutputHeat ) maxOutputHeat = temp; }
    public double getEvaporationConstant() { return evaporationRate; }
    public void setEvaporationConstant( double temp ) { if( temp >= 0 && temp <= 1 ) evaporationRate = temp; }
    public Object domEvaporationConstant() { return new Interval(0.0,1.0); }
    public double getDiffusionConstant() { return diffusionRate; }
    public void setDiffusionConstant( double temp ) { if( temp >= 0 && temp <= 1 ) diffusionRate = temp; }
    public Object domDiffusionConstant() { return new Interval(0.0, 1.0); }
    public double getRandomMovementProbability() { return randomMovementProbability; }
        
    public double[] getBugXPos() {
        try
            {
            double[] d = new double[bugs.length];
            for(int x=0;x<bugs.length;x++)
                {
                d[x] = ((Int2D)(buggrid.getObjectLocation(bugs[x]))).x;
                }
            return d;
            }
        catch (Exception e) { return new double[0]; }
        }
    
    public double[] getBugYPos() {
        try
            {
            double[] d = new double[bugs.length];
            for(int x=0;x<bugs.length;x++)
                {
                d[x] = ((Int2D)(buggrid.getObjectLocation(bugs[x]))).y;
                }
            return d;
            }
        catch (Exception e) { return new double[0]; }
        }


    public void setRandomMovementProbability( double t )
        {
        if (t >= 0 && t <= 1)
            {
            randomMovementProbability = t;
            for( int i = 0 ; i < bugCount ; i++ )
                if (bugs[i]!=null)
                    bugs[i].setRandomMovementProbability( randomMovementProbability );
            }
        }
    public Object domRandomMovementProbability() { return new Interval(0.0, 1.0); }
        
    public double getMaximumHeat() { return MAX_HEAT; }

    // we presume that no one relies on these DURING a simulation
    public int getGridHeight() { return gridHeight; }
    public void setGridHeight(int val) { if (val > 0) gridHeight = val; }
    public int getGridWidth() { return gridWidth; }
    public void setGridWidth(int val) { if (val > 0) gridWidth = val; }
    public int getBugCount() { return bugCount; }
    public void setBugCount(int val) { if (val >= 0) bugCount = val; }
    
    public DoubleGrid2D valgrid;
  //  public DoubleGrid2D valgrid2;
    public SparseGrid2D buggrid;
    

    /** Creates a HeatBugs simulation with the given random number seed. */
    public HeatBugs(long seed)
        {
        this(seed, 10000, 10000, 1000000);
        }
        
    public HeatBugs(long seed, int width, int height, int count)
        {
        super(seed);
        gridWidth = width; gridHeight = height; bugCount = count;
        createGrids();
        }

    protected void createGrids()
        {
        bugs = new HeatBug[bugCount];
        valgrid = new DoubleGrid2D(gridWidth, gridHeight,0);
   //     valgrid2 = new DoubleGrid2D(gridWidth, gridHeight, 0);
        buggrid = new SparseGrid2D(gridWidth, gridHeight);      
        }
    
 //   ThreadedDiffuser diffuser = null;
        
    /** Resets and starts a simulation */
    public void start()
        {
        super.start();  // clear out the schedule
    
		long locationSize = bugCount*2*Sizeof.INT;
		locationPointer = new Pointer();
		cudaHostAlloc(locationPointer, locationSize, cudaHostAllocPortable);
		location = locationPointer.getByteBuffer(0, locationSize).order(ByteOrder.nativeOrder()).asIntBuffer();
		
		
		
		idealTemp = new double[bugCount];
		heatOutput = new double[bugCount];    
        // make new grids
        createGrids();
        
        
        
        
        
		System.out.println("num Agent " + bugCount + "  size "+ gridWidth+" x " + gridHeight);
        // Schedule the heat bugs -- we could instead use a RandomSequence, which would be faster
        // But we spend no more than 3% of our total runtime in the scheduler max, so it's not worthwhile
        for(int x=0;x<bugCount;x++)
            {
			idealTemp[x] = random.nextDouble() * (maxIdealTemp - minIdealTemp) + minIdealTemp;
			heatOutput[x] = random.nextDouble() * (maxOutputHeat - minOutputHeat) + minOutputHeat;

     
            bugs[x] = new HeatBug(idealTemp[x],heatOutput[x],randomMovementProbability);           
            bugs[x].index = x;
            bugs[x].TheHeatBug = this;  
                    
            bugs[x].setLocation(new Int2D (random.nextInt(gridWidth),random.nextInt(gridHeight)));


            buggrid.InitObjectLocation(bugs[x],bugs[x].getLocation().x,bugs[x].getLocation().y);           
          
            }
 
         schedule.currentPhase = 2;               
                            

        schedule.scheduleRepeating(schedule.EPOCH, 1, new Steppable()
		{
			public void step(SimState state) { return; }
		});
  
  
  
        AsynchronousSteppable s = new AsynchronousSteppable()
		{
			boolean shouldQuit = false;
			int gpu_step = 0;
			int schedule_gpuSteps = 0;

			Object[] lock = new Object[0]; // an array is a unique, serializable object

			protected void run(boolean resuming)
			{
				boolean quit = false;
				if (!resuming)
				{	
					cuda_init();
					
					schedule_gpuSteps = schedule.getgpuSteps();
					
					System.out.println("Async agent started");

				}
				else // (resuming)
				{
					//carefully check here

					cuda_init();
					
					schedule_gpuSteps = schedule.getgpuSteps();
					
					System.out.println("Async agent resumed");
				}
				while(!quit)
				{
					if(schedule.currentPhase == 2)  //GPU phases
					{
						
						context.GPUHeatBug();
						
						context.eventRecord();
						context.eventSync();
						context.GPUDiffuse();			
						
						context.eventRecord();
						context.eventSync();

						gpu_step ++;
						if(gpu_step == schedule_gpuSteps) //time to copy back and wake up mason's Schedule
						{	
								context.copy_back(locationPointer, valgrid.DataPointer);
								gpu_step = 0;
								schedule.return_to_CPU();
						 }

						context.sync_data_peer();

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
    
    public void stop()
        {
    //    if (diffuser != null) diffuser.cleanup();
     //   diffuser = null;
        }
    
    public void cuda_init()
    {
               JCudaDriver.setExceptionsEnabled(true);
                JCuda.setExceptionsEnabled(true);
                JCurand.setExceptionsEnabled(true);
				cudaDeviceReset();
				cuInit(0);
				
				context = new HeatBugsGPUContext(0,0);
				context.InitHeatBugsConstant(gridWidth,gridHeight, MAX_HEAT, randomMovementProbability, evaporationRate,diffusionRate, bugCount);
				context.InitModule(1234,bugCount,0);
				context.InitMemory(locationPointer, idealTemp, heatOutput);
				context.InitKernelParameters();
				
		
	}
    
    /** This little function calls Runtime.getRuntime().availableProcessors() if it's available,
        else returns 1.  That function is nonexistent in Java 1.3.1, but it exists in 1.4.x.
        So we're doing a little dance through the Reflection library to call the method tentatively!
        The value returned by Runtime is the number of available processors on the computer.  
        If you're only using 1.4.x, then all this is unnecessary -- you can just call
        Runtime.getRuntime().availableProcessors() instead. */
    public static int availableProcessors()
        {
        Runtime runtime = Runtime.getRuntime();
        try { return ((Integer)runtime.getClass().getMethod("availableProcessors", (Class[])null).
                invoke(runtime,(Object[])null)).intValue(); }
        catch (Exception e) { return 1; }  // a safe but sometimes wrong assumption!
        }
        
    
    
    public static void main(String[] args)
        {
        doLoop(HeatBugs.class, args);
        System.exit(0);
        }    
    }
    
    
    
    
    
