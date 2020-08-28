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

void scaleUp(uchar *image, uchar *newImage)
{
	const int rowLength = XSIZE * 3 * 2;

	// We don't want to divide by zero, so the first line is exluded from the loop
	for (int k = 0; k < 3; k++)
	{
		int pixelValue = image[k];

		newImage[k] = pixelValue;
		newImage[3 + k] = pixelValue;
		newImage[rowLength + k] = pixelValue;
		newImage[3 + rowLength + k] = pixelValue;
	}

	int length = XSIZE * YSIZE * 3;
	for (int i = 3; i < length; i += 3)
	{
		int rowNumber = i / (XSIZE * 3);
		int rows = rowNumber * rowLength;
		int scaledXIndex = i * 2;

		for (int k = 0; k < 3; k++)
		{
			int pixelValue = image[i + k];

			newImage[scaledXIndex + rows + k] = pixelValue;
			newImage[3 + scaledXIndex + rows + k] = pixelValue;
			newImage[scaledXIndex + rows + rowLength + k] = pixelValue;
			newImage[3 + scaledXIndex + rows + rowLength + k] = pixelValue;
		}
	}
}

int main()
{
	uchar *image = calloc(XSIZE * YSIZE * 3, 1); // Three uchars per pixel (RGB)
	readbmp("before.bmp", image);

	// Alter the image here

	revertColors(image);
	rotateImage(image);

	uchar *newImage = calloc(XSIZE * YSIZE * 3 * 4, 1); // Double size in both directions
	scaleUp(image, newImage);

	savebmp("after.bmp", newImage, XSIZE * 2, YSIZE * 2);
	free(image);
	free(newImage);
	return 0;
}
