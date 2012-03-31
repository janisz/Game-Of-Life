#include "common.h"
#include "cpu_bitmap.h"
#include "cpu_anim.h"

#define DIM 200
#define INDEX(x, y) ((x)+(y)*DIM)

// Wartoœci wymagane przez procedurê aktualizuj¹c¹
struct DataBlock {
    unsigned char   *dev_bitmap;
	unsigned char   *dev_pom_bitmap;
	CPUAnimBitmap  *bitmap;
};

__device__ int step( int i, int j, unsigned char *col ) {

	int aliveNeighbours = 0;
	if (i != 0 && j != 0)
		aliveNeighbours += (col[4*INDEX(i - 1,j - 1)]) ? 1 : 0;
	if (i != 0)
	{
		aliveNeighbours += (col[4*INDEX(i - 1,j)]) ? 1 : 0;
		if (j+1 < DIM)
		aliveNeighbours += (col[4*INDEX(i - 1,j + 1)]) ? 1 : 0;
	}
	if (j != 0)
	{
		if (i+1 < DIM)	
		aliveNeighbours += (col[4*INDEX(i + 1,j - 1)]) ? 1 : 0;
		aliveNeighbours += (col[4*INDEX(i,j - 1)]) ? 1 : 0;
	}
	if (i+1 < DIM && j+1<DIM)
		aliveNeighbours += (col[4*INDEX(i + 1,j + 1)]) ? 1 : 0;
	if (j+1 < DIM)
		aliveNeighbours += (col[4*INDEX(i,j + 1)]) ? 1 : 0;
	if (i+1 < DIM)	
		aliveNeighbours += (col[4*INDEX(i + 1,j)]) ? 1 : 0;

	if (col[4*INDEX(i,j)] && aliveNeighbours > 1 && aliveNeighbours < 4)
		return aliveNeighbours;
	if (!col[4*INDEX(i,j)] && aliveNeighbours > 2 && aliveNeighbours < 4)
		return aliveNeighbours;
	return 0;
 
}

__global__ void kernel( unsigned char *ptr, unsigned char *pom, int t ) {
    // Odwzorowanie z blockIdx na po³o¿enie piksela
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;

    // Obliczenie wartoœci dla tego miejsca
    int isAlive = step( x, y, ptr );
    pom[offset*4 + 0] = 255 * isAlive;	//Red
    pom[offset*4 + 1] = 80 * isAlive;	//Green
    pom[offset*4 + 2] = 30 * isAlive;	//Blue
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

void generate_frame( DataBlock *d, int ticks )
{
	dim3    grid(DIM,DIM);
	if (ticks == 1)
		setBoard<<<grid,1>>>( d->dev_bitmap );

    kernel<<<grid,1>>>( d->dev_bitmap, d->dev_pom_bitmap, ticks );

    HANDLE_ERROR( cudaMemcpy( d->dev_bitmap, d->dev_pom_bitmap,
                              d->bitmap->image_size(),
                              cudaMemcpyDeviceToDevice ) );

    HANDLE_ERROR( cudaMemcpy( d->bitmap->get_ptr(), d->dev_bitmap,
                              d->bitmap->image_size(),
                              cudaMemcpyDeviceToHost ) );          

}
// Zwolnienie pamiêci na GPU
void cleanup( DataBlock *d ) {
    HANDLE_ERROR( cudaFree( d->dev_bitmap ) ); 
	HANDLE_ERROR( cudaFree( d->dev_pom_bitmap ) ); 
}



int main( void ) {
    DataBlock   data;
//    CPUBitmap bitmap( DIM, DIM, &data );
	CPUAnimBitmap  bitmap( DIM, DIM, &data );
    unsigned char    *dev_bitmap;
	unsigned char    *dev_pom_bitmap;
	data.bitmap = &bitmap;
    HANDLE_ERROR( cudaMalloc( (void**)&dev_bitmap, bitmap.image_size() ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_pom_bitmap, bitmap.image_size() ) );
    data.dev_bitmap = dev_bitmap;
	data.dev_pom_bitmap = dev_pom_bitmap;
       
	
    bitmap.anim_and_exit( (void (*)(void*,int))generate_frame,
                          (void (*)(void*))cleanup );
}

