#ifndef MAT_H
#define MAT_H

class Matrixsz
{
public:

/* Member Data */ 
    int width; 
    int height; 
    int my_type; 
    float* elements; 

/* Constructors */ 

        Matrixsz (const int w, const int h, const int type = 0){
        width = w;
        height = h;
       my_type = type; //Matrix knows if it's CPU or GPU
       if(type == 0)
            elements = new float[width*height];
        else
            cudaMalloc(&elements, width*height*sizeof(float));
            elements = new float[width*height];

    }

     //SZZ trying to add init function
    /*void init (const int w, const int h, const int type = 0){
        width = w; 
        height = h;
        my_type = type; //Matrix knows if it's CPU or GPU 
      //  if(type == 0)
        //    elements = new float[width*height];
       // else
            //cudaMalloc(&elements, width*height*sizeof(float)); 
	 //   elements = new float[width*height];
 
    }*/


/* member functions */ //SZZ diaable the the cudaMemcpy function for cdp 
    
    void load(const Matrixsz old_matrix, const int dir = 0){
        size_t size = width*height*sizeof(float);
        if(dir == 0){ //CPU copy
            memcpy(elements, old_matrix.elements, size); 
        }
        else if(dir == 1){ //GPU copy host to device
            cudaMemcpy(elements, old_matrix.elements, size, cudaMemcpyHostToDevice);  
        }
    }

    void cpu_deallocate(){
        delete elements; 
    }

    void gpu_deallocate(){
        cudaFree(elements); //Do not use cudaFree 
    }
};
#endif
