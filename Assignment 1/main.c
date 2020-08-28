#include <stdlib.h>
#include <stdio.h>
#include "bitmap.h"
#include "bitmap.c"

#define XSIZE 2560 // Size of before image
#define YSIZE 2048

// Reverts each pixel
void revertColors(uchar *image)
{
	for (int i = 0; i < XSIZE * YSIZE * 3; i++)
	{
		image[i] = 255 - image[i];
	}
}

// Rotates the image 180 degrees
void rotateImage(uchar *image)
{
	int length = XSIZE * YSIZE * 3;
	for (int i = 0; i < length / 2; i += 3)
	{
		// We modify the pixels RGB-blockwise such that RED matches RED, GREEN matches GREEN and so on.
		for (int k = 0; k < 3; k++)
		{
			uchar temp = image[i + k];
			image[i + k] = image[length - i + k];
			image[length - i + k] = temp;
		}
	}
}

void scaleUp(uchar *image)
{
}

int main()
{
	uchar *image = calloc(XSIZE * YSIZE * 3, 1); // Three uchars per pixel (RGB)
	readbmp("before.bmp", image);

	// Alter the image here

	revertColors(image);
	rotateImage(image);
	scaleUp(image);

	// Stop altering

	savebmp("after.bmp", image, XSIZE, YSIZE);
	free(image);
	return 0;
}
