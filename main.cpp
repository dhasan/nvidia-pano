#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <assert.h>
//#include <helper_string.h>  // helper for shared functions common to CUDA Samples

// CUDA and CUBLAS functions
//#include <helper_functions.h>
//#include <helper_cuda.h>

#define  IDX2C(i,j,ld) (((j)*(ld))+( i ))
#define  CUDART_PI_F 3.141592654f 

#define SETMATRIX3X3(data,a11,a12,a13,a21,a22,a23,a31,a32,a33)  \
            data[0]=a11;data[1]=a12;data[2]=a13;                \
            data[3]=a21;data[4]=a22;data[5]=a23;                \
            data[6]=a31;data[7]=a32;data[8]=a33;


#define SETMATRIX4X4(data,a11,a12,a13,a14,a21,a22,a23,a24,a31,a32,a33,a34,a41,a42,a43,a44)  \
            data[0]=a11;data[1]=a12;data[2]=a13;data[3]=a14;    \
            data[4]=a21;data[5]=a22;data[6]=a23;data[7]=a24;    \
            data[8]=a31;data[9]=a32;data[10]=a33;data[11]=a34;  \
            data[12]=a41;data[13]=a42;data[14]=a43;data[15]=a44; 

#define SETVECTOR4(data,x,y,z,w) data[0]=x;data[1]=y;data[2]=z;data[3]=w;

struct plane {

    int id;
    char *name;

    char *devname;
    int fd;

    /*Plane definition*/
    float4 plane;

    /*Set plane boarders, to define a rectangle*/
    float2 plane_min_max_x;
    float2 plane_min_max_y;
    float2 plane_min_max_z;

    /*Transpose origin to top left corner*/
    float3 o_transp;

    /*Rotate origin x- from left to right, y- from top to bottom, z- projection on plane is dot!*/
    float3 o_rotate;
    
    /*From now on we are in 2D space*/
    /*Scale plane to source image PoI*/
    float3 scale;

    /*Set offset from */
    float3 offset;
};

int main(){
    
    cudaError_t  cudaStat;                      //  cudaMalloc  status
    cublasStatus_t  stat;                //  CUBLAS  functions  status
    cublasHandle_t  handle; 
    
    int i,j;
    struct plane *p1;

    float *a_matrix;
    float *b_matrix;
    float *c_matrix;

    float *dev_a_matrix;
    float *dev_b_matrix;
    float *dev_c_matrix;

    float *in_vector;
    float *dev_in_vector;
    float *out_vector;
    float *dev_out_vector;

    p1 = (struct plane*)malloc(sizeof(struct plane));

    /*Define plane and constants for interception calculation*/
    p1->plane.x = 0.0f;
    p1->plane.y = 0.0f;
    p1->plane.z = 1.0f;
    p1->plane.w = -1.0f;
    

    p1->plane_min_max_x.x = -1.0f;
    p1->plane_min_max_x.y =  1.0f;
    p1->plane_min_max_y.x = -1.0f;
    p1->plane_min_max_y.y =  1.0f;
    p1->plane_min_max_z.x =  0.0f;
    p1->plane_min_max_z.y =  0.0f;

    p1->o_transp.x =  1.0f;
    p1->o_transp.y = -1.0f;
    p1->o_transp.z = -1.0f;

    p1->o_rotate.x = CUDART_PI_F;    //180 degree
    p1->o_rotate.y = 0.0f;
    p1->o_rotate.z = 0.0f;

    p1->scale.x = 500.0f;
    p1->scale.y = 500.0f;
    p1->scale.z = 1.0f;

    p1->offset.x = 100.0f;
    p1->offset.y = 100.0f;
    p1->offset.z = 0.0f;

    printf("Constands set..\n");

    unsigned int block_size = 16;
    unsigned int grid_size = 160;
  
    dim3 threads(block_size, block_size);
    dim3 grid(grid_size / threads.x, grid_size / threads.y);

    printf("Gird and Block size set..\n");
    stat = cublasCreate (&handle);

    printf("cublas handle created\n");
    /*Calculate affine transformation matrix*/
    a_matrix = (float *)malloc(4*4*sizeof(float));
    b_matrix = (float *)malloc(4*4*sizeof(float));
    c_matrix = (float *)malloc(4*4*sizeof(float));

    printf("CPU allocation is done\n");
    //A=transp
    SETMATRIX4X4(a_matrix,  1.0f, 0.0f, 0.0f, (p1->o_transp.x), 
                            0.0f, 1.0f, 0.0f, (p1->o_transp.y),
                            0.0f, 0.0f, 1.0f, (p1->o_transp.z),
                            0.0f, 0.0f, 0.0f, 1.0f )

    printf("Matrix defined..\n");

    //1. Rotate along X axis
    //B=rotate(x)
    SETMATRIX4X4(b_matrix,  1.0f, 0.0f,                 0.0f,                   0.0f, 
                            0.0f, cos(p1->o_rotate.x),  0-sin(p1->o_rotate.x),   0.0f,
                            0.0f, sin(p1->o_rotate.x),  cos(p1->o_rotate.x),    0.0f,
                            0.0f, 0.0f,                 0.0f,                   1.0f )


    printf("Matrix defined..\n");

    //Clear resulting matrix, do I need this?
    SETMATRIX4X4(c_matrix,  1.0f, 0.0f, 0.0f, 0.0f, 
                            0.0f, 1.0f, 0.0f, 0.0f,
                            0.0f, 0.0f, 1.0f, 0.0f,
                            0.0f, 0.0f, 0.0f, 1.0f )
    printf("Matrix defined..\n");


    cudaStat = cudaMalloc ((void **)&dev_a_matrix ,4*4*sizeof (*a_matrix));
    cudaStat = cudaMalloc ((void **)&dev_b_matrix ,4*4*sizeof (*b_matrix));
    //Do I need to clear?
    cudaStat = cudaMalloc ((void **)&dev_c_matrix ,4*4*sizeof (*c_matrix));

    stat = cublasSetMatrix(4,4,sizeof (*a_matrix),a_matrix,4,dev_a_matrix ,4);
    stat = cublasSetMatrix(4,4,sizeof (*b_matrix),b_matrix,4,dev_b_matrix ,4);
    stat = cublasSetMatrix(4,4,sizeof (*c_matrix),c_matrix,4,dev_c_matrix ,4);

    float al = 1.0f;
    float bet = 0.0f;

    //C=A*B
    stat=cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,4,4,4,&al,dev_a_matrix,4,dev_b_matrix,4,&bet,dev_c_matrix,4);
    printf("Matrix multipiled on GPU..\n");


    SETMATRIX4X4(b_matrix,  cos(p1->o_rotate.y),    0.0f,   sin(p1->o_rotate.y),    0.0f, 
                            0.0f,                   1.0f,   0.0f,                   0.0f,
                            -sin(p1->o_rotate.y),   0.0f,   cos(p1->o_rotate.y),    0.0f,
                            0.0f,                   0.0f,   0.0f,                   1.0f )
    printf("Matrix defined..\n");
    //A=C*B
    stat = cublasSetMatrix(4,4,sizeof (*b_matrix),b_matrix,4,dev_b_matrix ,4);
    printf("Matrix copied to GPU..\n");
    stat=cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,4,4,4,&al,dev_c_matrix,4,dev_b_matrix,4,&bet,dev_a_matrix,4);
    printf("Matrix multipiled on GPU..\n");
    

    SETMATRIX4X4(b_matrix,  cos(p1->o_rotate.z),    -sin(p1->o_rotate.z),   0.0f,   0.0f, 
                            sin(p1->o_rotate.z),    cos(p1->o_rotate.z),    0.0f,   0.0f,
                            0.0f,                   0.0f,                   1.0f,   0.0f,
                            0.0f,                   0.0f,                   0.0f,   1.0f )
    printf("Matrix defined..\n");
    //C=A*B
    stat = cublasSetMatrix(4,4,sizeof (*b_matrix),b_matrix,4,dev_b_matrix ,4);
    printf("Matrix copied to GPU..\n");
    stat=cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,4,4,4,&al,dev_a_matrix,4,dev_b_matrix,4,&bet,dev_c_matrix,4);
    printf("Matrix multipiled on GPU..\n");


    SETMATRIX4X4(b_matrix,  p1->scale.x,    0.0f,           0.0f,           0.0f, 
                            0.0f,           p1->scale.y,    0.0f,           0.0f,
                            0.0f,           0.0f,           p1->scale.z,    0.0f,
                            0.0f,           0.0f,           0.0f,           1.0f )
    printf("Matrix defined..\n");

    stat = cublasSetMatrix(4,4,sizeof (*b_matrix),b_matrix,4,dev_b_matrix ,4);
    printf("Matrix copied to GPU..\n");
    stat=cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,4,4,4,&al,dev_c_matrix,4,dev_b_matrix,4,&bet,dev_a_matrix,4);
    printf("Matrix multipiled on GPU..\n");


    SETMATRIX4X4(b_matrix,  1.0f,   0.0f,   0.0f,   p1->offset.x, 
                            0.0f,   1.0f,   0.0f,   p1->offset.y,
                            0.0f,   0.0f,   1.0f,   p1->offset.z,
                            0.0f,   0.0f,   0.0f,   1.0f )
    printf("Matrix defined..\n");

    stat = cublasSetMatrix(4,4,sizeof (*b_matrix),b_matrix,4,dev_b_matrix ,4);
    printf("Matrix copied to GPU..\n");
    stat=cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,4,4,4,&al,dev_a_matrix,4,dev_b_matrix,4,&bet,dev_c_matrix,4);
    printf("Matrix multipiled on GPU..\n");
    //A - output

    for(j=0;j<4;j++){
        for(i=0;i<4;i++){                   
            a_matrix[IDX2C(i,j,4)]=0.0;      
        }                                         
    }

    stat=cublasGetMatrix(4,4,sizeof (*c_matrix),dev_c_matrix ,4,c_matrix,4);
    printf("Transformation matrix:\n");
    for(j=0;j<4;j++){
        printf("\t");
        for(i=0;i<4;i++){
            printf("%7.0f",c_matrix[IDX2C(i,j,4)]);   // print c after  Sgemm
        }
        printf("\n");
    } 
    printf("\n");

//No lets make some tests
    in_vector = (float *)malloc(4*sizeof(float));
    out_vector = (float *)malloc(4*sizeof(float));
    SETVECTOR4(in_vector,  0.0f,0.0f,1.0f,1.0f)
    SETVECTOR4(out_vector, 0.0f,0.0f,0.0f,1.0f)

    cudaStat=cudaMalloc((void **)&dev_in_vector ,4*1*sizeof (*in_vector)); 
    cudaStat=cudaMalloc((void **)&dev_out_vector ,4*1*sizeof (*out_vector)); 
    stat = cublasSetVector(4,sizeof (*in_vector),in_vector,1,dev_in_vector ,1);
    stat = cublasSetVector(4,sizeof (*out_vector),out_vector,1,dev_out_vector ,1);

    stat=cublasSgemv(handle,CUBLAS_OP_T,4,4,&al,dev_c_matrix,4,dev_in_vector,1,&bet,dev_out_vector,1);
    cudaThreadSynchronize();

    printf("stat: %d\n",stat);
    stat=cublasGetVector(4,sizeof (*out_vector),dev_out_vector ,1,out_vector,1);
    
    printf("out vector:\n");
    printf("\t");
    for(i=0;i<4;i++){
        printf("%7.0f",out_vector[i]);   // print c after  Sgemm
    }
    printf("\n");

    float x,y,z;
    float theta,phi;
    phi = CUDART_PI_F/4.0f;
    for(theta=-CUDART_PI_F/4.0f;theta<CUDART_PI_F/4.0f;theta+=(CUDART_PI_F/450.0f)){
         x = sin(phi)*cos(theta);
         y = sin(phi)*sin(theta);
         z = cos(phi);

        // x = cos(phi) * cos(theta); 
        // y = sin(phi); 
        // z = cos(phi) * sin(theta);

        SETVECTOR4(in_vector, x,y,z,1.0f)
        stat = cublasSetVector(4,sizeof (*in_vector),in_vector,1,dev_in_vector ,1);
        stat=cublasSgemv(handle,CUBLAS_OP_T,4,4,&al,dev_c_matrix,4,dev_in_vector,1,&bet,dev_out_vector,1);
        stat=cublasGetVector(4,sizeof (*out_vector),dev_out_vector ,1,out_vector,1);
        printf("%7.2f %7.2f %7.2f\n",out_vector[0],out_vector[1],out_vector[2]); 
    }

    cudaFree(dev_a_matrix);                             // free  device  memory
    cudaFree(dev_b_matrix);                             // free  device  memory
    cudaFree(dev_c_matrix);                             // free  device  memory
    cudaFree(dev_in_vector); 
    cudaFree(dev_out_vector);
    cublasDestroy(handle );               //  destroy  CUBLAS  context
    free(a_matrix);                                       // free  host  memory
    free(b_matrix);                                       // free  host  memory
    free(c_matrix);
    free(in_vector);
    free(out_vector);  


    return 0;
}