/*
  Copyright 2006 by Sean Luke and George Mason University
  Licensed under the Academic Free License version 3.0
  See the file "LICENSE" for more information
*/

package sim.app.students;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.runtime.JCuda.*;

import jcuda.*;
import jcuda.driver.*;
import java.util.Arrays;
import jcuda.jcurand.*;
import jcuda.runtime.JCuda;

import java.io.*;
import java.nio.ByteOrder;
import java.nio.ByteBuffer; //byte buffer for better copy time, but the code will be more complicated
import java.nio.DoubleBuffer;



import sim.engine.*;
import sim.util.*;
import sim.field.continuous.*;
import sim.field.network.*;






public class Students extends SimState
    {
    private static final long serialVersionUID = 1;

    public Continuous2D yard = new Continuous2D(1.0,100,100);
    
//	public double[] loc_x;
//	public double[] loc_y;
	
	public Pointer locationPointer;
	public DoubleBuffer location;

	public Pointer agitationPointer;
	public DoubleBuffer agitation;

	
//    public double[] agitation;
    
    
    StudentsGPUContext Devices[];
    
    public double TEMPERING_CUT_DOWN = 0.99;
    public double TEMPERING_INITIAL_RANDOM_MULTIPLIER = 10.0;
    public boolean tempering = false;
    public boolean isTempering() { return tempering; }
    public void setTempering(boolean val) { tempering = val; }
        
    public int numStudents = 500000;

    double forceToSchoolMultiplier = 0.01;
    double randomMultiplier = 0.1;
	
	public int numDevices =4;

	public int MinimumLikeAndDislike = 10;

    public int getNumStudents() { return numStudents; }
    public void setNumStudents(int val) { if (val > 0) numStudents = val; }

    public double getForceToSchoolMultiplier() { return forceToSchoolMultiplier; }
    public void setForceToSchoolMultiplier(double val) { if (forceToSchoolMultiplier >= 0.0) forceToSchoolMultiplier = val; }

    public double getRandomMultiplier() { return randomMultiplier; }
    public void setRandomMultiplier(double val) { if (randomMultiplier >= 0.0) randomMultiplier = val; }
    public Object domRandomMultiplier() { return new sim.util.Interval(0.0, 100.0); }

    public double[] getAgitationDistribution()
        {
        Bag students = buddies.getAllNodes();
        double[] distro = new double[students.numObjs];
        int len = students.size();
        for(int i = 0; i < len; i++)
            distro[i] = ((Student)(students.get(i))).getAgitation();
        return distro;
        }

    public Network buddies = new Network(false);

    public Students(long seed)
        {
        super(seed);
        }

    public void start()
        {
        super.start();
        
        // add the tempering agent
        if (tempering)
            {
            randomMultiplier = TEMPERING_INITIAL_RANDOM_MULTIPLIER;
            schedule.scheduleRepeating(schedule.EPOCH, 1, new Steppable() 
                { public void step(SimState state) { if (tempering) randomMultiplier *= TEMPERING_CUT_DOWN; } });
            }
                
        // clear the yard
        yard.clear();

        // clear the buddies
        buddies.clear();

        
        long locationSize = (numStudents*2)*Sizeof.DOUBLE;
		long agitationSize = (numStudents)*Sizeof.DOUBLE;
 
     
		locationPointer = new Pointer();
		agitationPointer = new Pointer();
		
		cudaHostAlloc(locationPointer, locationSize, cudaHostAllocPortable); //multiple devices, multiple contexts
		cudaHostAlloc(agitationPointer, agitationSize, cudaHostAllocPortable); 
			
		location = locationPointer.getByteBuffer(0, locationSize).order(ByteOrder.nativeOrder()).asDoubleBuffer();
		agitation = agitationPointer.getByteBuffer(0, agitationSize).order(ByteOrder.nativeOrder()).asDoubleBuffer();

        int numElements = numStudents/numDevices;
        // add some students to the yard
        for(int i = 0; i < numStudents; i++)
            {
            Student student = new Student();
  			student.students = this;          
			student.index = i;
    
            student.setLocation(new Double2D (yard.getWidth() * 0.5 + random.nextDouble() - 0.5, yard.getHeight() * 0.5 + random.nextDouble() - 0.5 ));

            yard.initObjectLocation(student, student.getLocation());

            buddies.addNode(student);
   
            }
            
       
        // define like/dislike relationships
        Bag students = buddies.getAllNodes();
        for(int i = 0; i < students.size(); i++)
        {
            Object student = students.get(i);
            
            for(int j = 0; j< MinimumLikeAndDislike; j++)
            {
            // who does he like?
				Object studentB = null;
				do
					{
					studentB = students.get(random.nextInt(students.numObjs));
					} while (student == studentB);
				double buddiness = random.nextDouble();
				buddies.addEdge(student, studentB, new Double(buddiness));

				// who does he dislike?
				do
					{
					studentB = students.get(random.nextInt(students.numObjs));
					} while (student == studentB);
				buddiness = random.nextDouble();
				buddies.addEdge(student, studentB, new Double( -buddiness));
			}
        }
            
		
//initial edge array
       	schedule.scheduleRepeating(schedule.EPOCH, 1, new Steppable()
		{
			public void step(SimState state) { return; }
		});   

		schedule.currentPhase = 2;
		System.out.println("students " + numStudents) ;


        AsynchronousSteppable s = new AsynchronousSteppable()
		{
			boolean shouldQuit = false;
			int gpu_step = 0;
			int schedule_gpuSteps =0;
		//	long my_start_time = 0;
			long start_time = 0;
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
					cuda_init();
					schedule_gpuSteps = schedule.getgpuSteps();
					System.out.println("Async agent resumed");
				}
				while(!quit)
				{
					if(schedule.currentPhase == 2)  //GPU phases
					{

					for(int i =0; i<numDevices ;i++)
					{
							Devices[i].launchKernel();
							Devices[i].eventRecord();
					}
					
					for(int i =0; i<numDevices ;i++)
							Devices[i].eventSync();
							
							gpu_step ++;
							if(gpu_step == schedule_gpuSteps) //time to copy back and wake up mason's Schedule
							{
								for(int i =0; i<numDevices ;i++)
									Devices[i].copy_back(locationPointer,agitationPointer);						
								gpu_step = 0;
								schedule.return_to_CPU();
							}		
							
					for(int i =0; i<numDevices ;i++)
					{
							Devices[i].sync_data_peer(Devices);
							Devices[i].eventRecord();
					}						
					for(int i =0; i<numDevices ;i++)
					{
							Devices[i].eventSync();	
							Devices[i].sync_internal_data();
							Devices[i].eventRecord();		
					}
					
					for(int i =0; i<numDevices ;i++)
							Devices[i].eventSync();
				
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
        doLoop(Students.class, args);
        System.exit(0);
        }
        
        
        
        
        
        public void cuda_init() 
        {

				JCudaDriver.setExceptionsEnabled(true);
                JCuda.setExceptionsEnabled(true);
                JCurand.setExceptionsEnabled(true);
				cudaDeviceReset();
				cuInit(0);
		//		int numElements = numStudents;
				Devices = new StudentsGPUContext[numDevices];  //array of GPU contexts, can be applied to multiple GPUs
				int numElements = numStudents/numDevices;
		//		int lastNumElements = 0;
				
				for(int i =0; i<numDevices ;i++)
				{
					Devices[i] = new StudentsGPUContext(i,i);
					Devices[i].InitStudentsConstant(forceToSchoolMultiplier ,randomMultiplier, yard.getWidth(),yard.getHeight());
					Devices[i].InitModule(1234, numElements, numStudents, numDevices); 
					Devices[i].InitMemory(locationPointer, buddies);
					Devices[i].InitKernelParameters();	
				}
		}    
        
        
        
        
            
    }
