#include <stdlib.h>
#include <stdio.h>

typedef struct
{
    unsigned char b;
    unsigned char g;
    unsigned char r;
} pixel;

pixel im[9] = {
    (pixel){1, 1, 1}, (pixel){2, 2, 2}, (pixel){3, 3, 3},
    (pixel){4, 4, 4}, (pixel){5, 5, 5}, (pixel){6, 6, 6},
    (pixel){7, 7, 7}, (pixel){8, 8, 8}, (pixel){9, 9, 9}

    // TODO create the col array to check against, 9x9 array

};
int filter[9] = {0, 1, 0,
                 1, 2, 1,
                 0, 1, 0};
pixel answer[9] = {
    (pixel){8, 8, 8},    // 2+2+4
    (pixel){13, 13, 13}, // 4+1+5+3
    (pixel){14, 14, 14}, // 6+2+5
    (pixel){21, 21, 21}, // 8+1+5+7
    (pixel){30, 30, 30}, // 10+2+6+4+8
    (pixel){29, 29, 29}, // 12+3+5+9
    (pixel){26, 26, 26}, //  14+4+8
    (pixel){37, 37, 36}, // 16+7+5+9
    (pixel){32, 32, 32}  // 18 + 8 + 6
};

// im2col and col2im taken from https://github.com/pluskid/Mocha.jl/blob/master/deps/im2col.cpp#L7
// They only work with one channel (the RGB channel is "one" for the pixel struct).
// Furthemore, we assume that we use a valid padding (input dim = output dim) and 1 stride.

void im2col(pixel *img, pixel *col, int width, int height, int filterDim)
{
    int kernel_h = filterDim,  // Assuming square kernel
        kernel_w = filterDim,  // Assuming square kernel
        pad_h = filterDim / 2, // Same padding
        pad_w = filterDim / 2; // Same padding
    int height_col = (height + 2 * pad_h - kernel_h) + 1;
    int width_col = (width + 2 * pad_w - kernel_w) + 1;
    int channels_col = kernel_h * kernel_w;

    for (int c = 0; c < channels_col; ++c)
    {
        int w_offset = c % kernel_w;
        int h_offset = (c / kernel_w) % kernel_h;
        int c_im = c / (kernel_h * kernel_w);

        for (int h = 0; h < height_col; ++h)
        {
            for (int w = 0; w < width_col; ++w)
            {
                int h_pad = h - pad_h + h_offset;
                int w_pad = w - pad_w + w_offset;
                int index_col = (c * height_col + h) * width_col + w;
                int index_im = (c_im * height + h_pad) * width + w_pad;
                if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
                {
                    col[index_col] = img[index_im];
                }
                else
                {
                    col[index_col] = (pixel){0, 0, 0};
                }
            }
        }
    }
}

void col2im(pixel *col, pixel *img, int width, int height, int filterDim)
{
    int kernel_h = filterDim,  // Assuming square kernel
        kernel_w = filterDim,  // Assuming square kernel
        pad_h = filterDim / 2, // Same padding
        pad_w = filterDim / 2; // Same padding
    int height_col = (height + 2 * pad_h - kernel_h) + 1;
    int width_col = (width + 2 * pad_w - kernel_w) + 1;
    int channels_col = kernel_h * kernel_w;

    // Fill the img array with black pixels, I do not remember the memcpy function for this. May fix later..
    for (int i = 0; i < width * height; i++)
    {
        img[i] = (pixel){0, 0, 0};
    }

    for (int c = 0; c < channels_col; ++c)
    {
        int w_offset = c % kernel_w;
        int h_offset = (c / kernel_w) % kernel_h;
        int c_im = c / (kernel_h * kernel_w);

        for (int h = 0; h < height_col; ++h)
        {
            for (int w = 0; w < width_col; ++w)
            {
                int h_pad = h - pad_h + h_offset;
                int w_pad = w - pad_w + w_offset;
                int index_col = (c * height_col + h) * width_col + w;
                int index_im = (c_im * height + h_pad) * width + w_pad;
                if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
                {
                    // There is a lot of redundant value setting here, but it works :)
                    img[index_im].r = col[index_col].r;
                    img[index_im].g = col[index_col].g;
                    img[index_im].b = col[index_col].b;
                }
            }
        }
    }
}

int main(int argc, char const *argv[])
{
    pixel *col = malloc(9 * 9 * sizeof(pixel));
    pixel *im_from_col = malloc(9 * sizeof(pixel));

    im2col(im, col, 3, 3, 3);

    // Good
    // for (int i = 0; i < 9 * 9; i++)
    // {
    //     if (i % 9 == 0)
    //     {
    //         printf("\n");
    //     }
    //     printf("{%d, %d, %d}, ", col[i].r, col[i].g, col[i].b);
    // }

    col2im(col, im_from_col, 3, 3, 3);

    // for (size_t i = 0; i < 9; i++)
    // {
    //     if (i % 3 == 0)
    //     {
    //         printf("\n");
    //     }
    //     printf("{%d, %d, %d}, ", im_from_col[i].r, im_from_col[i].g, im_from_col[i].b);
    // }
    for (int i = 0; i < 9; i++)
    {
        if (im_from_col[i].r != im[i].r || im_from_col[i].g != im[i].g || im_from_col[i].b != im[i].b)
        {
            printf("not correct\n");
            exit(1);
        }
    }
    printf("OK!\n");
    return 0;
}
