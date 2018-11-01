#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "uwnet.h"

// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_maxpool_layer(layer l, matrix in)
{
    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    matrix out = make_matrix(in.rows, outw*outh*l.channels);

    // 6.1 - iterate over the input and fill in the output with max values
    // TODO: is this rite??

    l.in[0] = in;
    free_matrix(l.out[0]);
    l.out[0] = out;
    free_matrix(l.delta[0]);
    l.delta[0] = make_matrix(out.rows, out.cols);
    return out;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix prev_delta: error term for the previous layer
void backward_maxpool_layer(layer l, matrix prev_delta)
{
    matrix in    = l.in[0];
    matrix out   = l.out[0];
    matrix delta = l.delta[0];

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;

    // 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.
    
    //printf("delta.rows: %d\tdelta.cols: %d\tprev_delta.rows: %d\tprev_delta.cols: %d\n", delta.rows, delta.cols, prev_delta.rows, prev_delta.cols);

    int num; // data point number corresponding to 
    for (num = 0; num < delta.rows; num++) {

        float *inim = &in.data[num * in.cols]; // "input image"
        float *ditu = &delta.data[num * delta.cols]; // "delta image to update"
        float *pdsrc = &prev_delta.data[num * prev_delta.cols]; // "prev_delta source"

        for (int c = 0; c < l.channels; c++) {

            float *cinim = &inim[c * l.height * l.width]; // "channel input image"
            float *cditu = &ditu[c * l.height * l.width]; // "channel delta image to update"
            float *cpdsrc = &pdsrc[c * outw * outh]; // "channel prev_delta source"

            for (int pdx = 0; pdx < outh; pdx++) {
                for (int pdy = 0; pdy < outw; pdy++) {
                    
                    int out_index = pdx * outw + pdy; // index in output (small) matrix
                    int maxi = 0;

                    int xbase = pdx * l.stride;
                    int ybase = pdy * l.stride;

                    // search the "pool" for max value
                    for (int dx = 0; dx < l.stride; dx++) {
                        for (int dy = 0; dy < l.stride; dy++) {

                            // indices in delta (and in) array
                            int x = xbase + dx;
                            int y = ybase + dy;
                            
                            // coordinate assertions
                            assert(x >= 0);
                            assert(x < l.height);
                            assert(y >= 0);
                            assert(y < l.width);

                            int index = x * l.width + y;

                            // index assertions
                            assert(index >= 0);
                            assert(index < l.width * l.height);

                            if (cinim[index] > cinim[maxi]) {
                                maxi = index;
                            }
                        }
                    }

                    // assign that max value backwards
                    cditu[maxi] = cpdsrc[out_index];
                }
            }
        }
    }
}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer l, float rate, float momentum, float decay)
{
}

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(int w, int h, int c, int size, int stride)
{
    layer l = {0};
    l.width = w;
    l.height = h;
    l.channels = c;
    l.size = size;
    l.stride = stride;
    l.in = calloc(1, sizeof(matrix));
    l.out = calloc(1, sizeof(matrix));
    l.delta = calloc(1, sizeof(matrix));
    l.forward  = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update   = update_maxpool_layer;
    return l;
}

