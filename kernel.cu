#include "common.h"
#include "cpu_bitmap.h"

#define DIM 100
#define INDEX(x, y) ((x)+(y)*DIM)

__device__ int step( int i, int j, unsigned char *col ) {

	int aliveNeighbours = 0;
	if (i != 0 && j != 0)
		aliveNeighbours += (col[4*INDEX(i - 1,j - 1)]) ? 1 : 0;
	if (i != 0)
	{
		aliveNeighbours += (col[4*INDEX(i - 1,j)]) ? 1 : 0;
		aliveNeighbours += (col[4*INDEX(i - 1,j + 1)]) ? 1 : 0;
	}
	if (j != 0)
	{
		aliveNeighbours += (col[4*INDEX(i + 1,j - 1)]) ? 1 : 0;
		aliveNeighbours += (col[4*INDEX(i,j - 1)]) ? 1 : 0;
	}
	if (j + 1 < DIM)
	{
		aliveNeighbours += (col[4*INDEX(i,j + 1)]) ? 1 : 0;
		if (i + 1 < DIM)
			aliveNeighbours += (col[4*INDEX(i + 1,j + 1)]) ? 1 : 0;
	}
	if (i + 1 < DIM)
		aliveNeighbours += (col[4*INDEX(i + 1,j)]) ? 1 : 0;

	if (col[4*INDEX(i,j)] && aliveNeighbours > 1 && aliveNeighbours < 4)
		return 1;
	if (!col[4*INDEX(i,j)] && aliveNeighbours > 2 && aliveNeighbours < 4)
		return 1;
	return 0;
 
}

__global__ void kernel( unsigned char *ptr, unsigned char *pom ) {
    // Odwzorowanie z blockIdx na po³o¿enie piksela
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;

    // Obliczenie wartoœci dla tego miejsca
    int isAlive = step( x, y, ptr );
    pom[offset*4 + 0] = 255 * isAlive;	//Red
    pom[offset*4 + 1] = 255 * isAlive;	//Green
    pom[offset*4 + 2] = 255 * isAlive;	//Blue
    pom[offset*4 + 3] = 255 * isAlive;	//Alpha
}

//ustawia szachownice
__global__ void setBoard( unsigned char *ptr ) {
    // Odwzorowanie z blockIdx na po³o¿enie piksela
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;

    // Obliczenie wartoœci dla tego miejsca
    int isAlive = (offset % 2 + y % 2) %2;
    ptr[offset*4 + 0] = 255 * isAlive;	//Red
    ptr[offset*4 + 1] = 255 * isAlive;	//Green
    ptr[offset*4 + 2] = 255 * isAlive;	//Blue
    ptr[offset*4 + 3] = 255 * isAlive;
}

// Wartoœci wymagane przez procedurê aktualizuj¹c¹
struct DataBlock {
    unsigned char   *dev_bitmap;
};

int main( void ) {
    DataBlock   data;
    CPUBitmap bitmap( DIM, DIM, &data );
    unsigned char    *dev_bitmap;
	unsigned char    *dev_pom_bitmap;

    HANDLE_ERROR( cudaMalloc( (void**)&dev_bitmap, bitmap.image_size() ) );
	    HANDLE_ERROR( cudaMalloc( (void**)&dev_pom_bitmap, bitmap.image_size() ) );
    data.dev_bitmap = dev_bitmap;

    dim3    grid(DIM,DIM);
	setBoard<<<grid,1>>>( dev_bitmap );
	for (int i=0;i<26;i++)
	{
    kernel<<<grid,1>>>( dev_bitmap, dev_pom_bitmap );

    HANDLE_ERROR( cudaMemcpy( dev_bitmap, dev_pom_bitmap,
                              bitmap.image_size(),
                              cudaMemcpyDeviceToDevice ) );

    HANDLE_ERROR( cudaMemcpy( bitmap.get_ptr(), dev_bitmap,
                              bitmap.image_size(),
                              cudaMemcpyDeviceToHost ) );
	}                          
    HANDLE_ERROR( cudaFree( dev_bitmap ) );
    HANDLE_ERROR( cudaFree( dev_pom_bitmap ) );
	bitmap.Scale(10);
    bitmap.display_and_exit();
}

