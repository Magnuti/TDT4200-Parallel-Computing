#include <stdio.h>

int main()
{
    int x;  /* A normal integer*/
    int *p; /* A pointer to an integer ("*p" is an integer, so p
                       must be a pointer to an integer) */

    p = &x; /* Read it, "assign the address of x to p" */
    printf("Input an int: ");
    scanf("%d", &x);              /* Put a value in x, we could also use p here */
    printf("Value of p: %d", *p); /* Note the use of the * to get the value */
    printf("\nAddress of p: %d", p);
    printf("\nx=%d", x);
    getchar();
}