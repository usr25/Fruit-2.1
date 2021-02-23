//TODO: Swap asserts for ASSERTs
#include <assert.h>

#include "board.h"
#include "util.h"
#include "nnue.h"
#include "nnuearch.h"

// variables

static clipped_t clippedInput[kDimensionFT];
static clipped_t hiddenLayer1[kDimensionHidden];
static clipped_t hiddenLayer2[kDimensionHidden];

// functions

// get_idx()

const int get_idx(const int i, const int j, const int dim) {
    return j*kDimensionHidden+i;
}

// propagate_input()

static void propagate_input(const int16_t* __restrict__ input, const int stm,
    clipped_t* __restrict__ nextLayer,
    const weight_t* ws, const int32_t* bs)
{
    assert(stm == 1 || stm == 0);

    int32_t tmp[kDimensionHidden];
    for (int i = 0; i < kDimensionHidden; ++i)
        tmp[i] = bs[i];

    const int offset = (1^stm)*kHalfDimensionFT;
    const int offset2 = kHalfDimensionFT ^ offset;

    for (int i = 0; i < kDimensionFT; ++i)
        clippedInput[i] = clip(input[i]);

    int idx, ni;
    for (int i = 0; i < kHalfDimensionFT; ++i)
    {
        idx = i + offset;
        if (clippedInput[idx])
            for (int j = 0; j < kDimensionHidden; ++j)
                tmp[j] += clippedInput[idx]*ws[kDimensionHidden*i+j];

        idx = i + offset2;
        ni = i + kHalfDimensionFT;
        if (clippedInput[idx])
            for (int j = 0; j < kDimensionHidden; ++j)
                tmp[j] += clippedInput[idx]*ws[kDimensionHidden*ni+j];
    }

    for (unsigned i = 0; i < kDimensionHidden; i++)
        nextLayer[i] = clip64(tmp[i]);
}

// propagate()

static void propagate(const clipped_t* __restrict__ prevLayer, const int prevSize,
    clipped_t* __restrict__ nextLayer, const int nextSize,
    const weight_t* ws, const int32_t* bs)
{
    int32_t tmp[kDimensionHidden];
    for (int i = 0; i < kDimensionHidden; ++i)
        tmp[i] = bs[i];

    for (int i = 0; i < prevSize; ++i)
    {
        if (prevLayer[i])
            for (int j = 0; j < kDimensionHidden; ++j)
                tmp[j] += prevLayer[i]*ws[kDimensionHidden*i+j];
    }

    for (unsigned i = 0; i < kDimensionHidden; i++)
        nextLayer[i] = clip64(tmp[i]);
}

// output()

static int32_t output(const clipped_t* __restrict__ prevLayer,
    const weight_t* __restrict__ ws, int32_t out)
{
    for (int i = 0; i < kDimensionHidden; ++i)
        out += prevLayer[i]*ws[i];

    return out;
}

// evaluate()

int evaluate(const NNUE* nn, const board_t* b, int16_t* nInput)
{
    input_layer(nn, b, White, nInput + kHalfDimensionFT);
    input_layer(nn, b, Black, nInput);

    return evaluate_acc(nn, b, nInput);
}

// evaluate_acc()

//#define TEST_ACC

#ifdef TEST_ACC
static int32_t testInput[kDimensionFT];
#endif
int evaluate_acc(const NNUE* nn, const board_t* b, const int16_t* nInput)
{
    #ifdef TEST_ACC
    input_layer(nn, b, White, testInput + kHalfDimensionFT);
    input_layer(nn, b, Black, testInput);

    for (int i = 0; i < kDimensionFT; ++i)
        assert(testInput[i] == nInput[i]);
    #endif

    propagate_input(nInput, b->turn, hiddenLayer1, nn->weights1, nn->biases1);
    propagate(hiddenLayer1, dimensions[2], hiddenLayer2, dimensions[3], nn->weights2, nn->biases2);

    const int32_t out = output(hiddenLayer2, nn->outputW, *(nn->outputB));

    return out / FV_SCALE;
}

// perft_nnue()

//To test if the "Efficiently Updatable" part works correctly
void perft_nnue(board_t b, int depth)
{
#ifdef TEST_ACC
   list_t ls;
   undo_t undo;
   gen_legal_moves(&ls, &b);
   int move;

    if (depth == 0)
      return;

   NNUEChangeList q;
   q.idx = 0;

   for (int i = 0; i < ls.size; i++) {
      move = ls.move[i];

      determine_changes(move, &b, &q);
      move_do(&b, move, &undo);
      update_do(&q, move, &b);
      perft_nnue(b, depth-1);
      move_undo(&b, move, &undo);
      update_undo(&q, &b);
   }
#else
   fprintf(stderr, "Compile with -DTEST_ACC\n");
#endif
}