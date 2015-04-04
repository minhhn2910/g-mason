//return new Double2D(x * dist / old, y * dist / old); resize function

extern "C" __global__ void students(
									int numThreads,
									int begin_offset,
									double width,
									double height,
									double randomMultiplier,
									double forceToSchoolMultiplier,
									int *node_size,
					//				int ** index_list,
									double** edge_list,
									double* random,
									double* location_old,
									double* location,
									double* agitation
									)

{
//MAX_FORCE in student.java = 3.0;
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   if (index<numThreads)
    {
		int location_index = index + begin_offset;
		
        double local_agitation = 0.0;
        
        double me_x = location_old[location_index*2];
        double me_y = location_old[location_index*2+1];
        
        double dx = 0.0;
        double dy = 0.0;
        int len = node_size[index];
        for(int i = 0 ; i < len; i++)
            {
            
				double buddiness = edge_list[index][2*i+1];
				int his_index = (int)(edge_list[index][2*i]); 


				double him_x = location_old[his_index*2];
				double him_y = location_old[his_index*2+1];
		
				
				double	temp_dx =(him_x - me_x) * buddiness;
				double	temp_dy =(him_y - me_y) * buddiness;
				
				double distance = hypot(temp_dx,temp_dy);
				
				if (buddiness >= 0)  // the further I am from him the more I want to go to him
					{
						
						if (distance > 3.0)  // I'm far enough away
						{
							temp_dx = temp_dx*3.0/distance;
							temp_dy = temp_dy*3.0/distance;
							distance = 3.0;
						}
						
						local_agitation += distance;
					}
				else  // the nearer I am to him the more I want to get away from him, up to a limit
					{

						if (distance > 3.0)  // I'm far enough away
						{
							temp_dx = 0.0;
							temp_dy = 0.0;
							distance = 0.0;
						}
						else if (distance > 0)
						{
							double new_dist = 3.0 - distance;
							temp_dx = temp_dx*new_dist/distance;
							temp_dy = temp_dy*new_dist/distance;  // invert the distance
							distance = new_dist;
						}
						
						local_agitation += distance;
					}
					
				dx += temp_dx;
				dy += temp_dy;
            
            }
        

        // add in a vector to the "teacher" -- the center of the yard, so we don't go too far away
        
        dx += (width * 0.5 - me_x)*forceToSchoolMultiplier + randomMultiplier*(random[location_index*2] * 1.0 - 0.5) + me_x;
        dy += (height * 0.5 - me_y)*forceToSchoolMultiplier + randomMultiplier*(random[location_index*2+1] * 1.0 - 0.5) + me_y;
        
//output
        location[location_index*2] = dx;
        location[location_index*2 +1] = dy;
        agitation[location_index] = local_agitation;

		//loc_x[index] = len;

	}

}
