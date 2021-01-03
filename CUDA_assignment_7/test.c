#include <stdio.h>
#include <stdlib.h>

int main(int argc, char const *argv[])
{
    unsigned char *x = (unsigned char *)malloc(5 * sizeof(unsigned char));
    float *y = (float *)malloc(5 * sizeof(float));
    for (int i = 0; i < 5; i++)
    {
        x[i] = i;
    }

    for (int i = 0; i < 5; i++)
    {
        y[i] = (float)x[i];
    }

    for (int i = 0; i < 5; i++)
    {
        x[i] = (unsigned char)y[i];
    }

    for (int i = 0; i < 5; i++)
    {
        printf("%f\n", y[i]);
    }

    for (int i = 0; i < 5; i++)
    {
        printf("%d\n", x[i]);
    }

    return 0;
}
