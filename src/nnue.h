
// nnue.h

#ifndef NNUE_H
#define NNUE_H

#include <stdint.h>

#define weight_t int8_t
#define clipped_t int8_t

typedef struct
{
    int16_t* ftBiases;
    int16_t* ftWeights;

    weight_t weights1[32*512];
    weight_t weights2[32*32];
    weight_t outputW[1*32];

    int32_t biases1[32];
    int32_t biases2[32];
    int32_t outputB[1];
} NNUE;

typedef struct
{
    int piece;
    int colour;
    int sqr;
    int appears;
} NNUEChange;

typedef struct
{
    NNUEChange changes[4];
    int idx;
} NNUEChangeList;

enum
{
    kHalfDimensionFT = 256,
    kDimensionFT = 512,
    kDimensionHidden = 32,
    kInputDimensionsFT = 41024
};

enum
{
    FV_SCALE = 16,
};

const int NN_KING = -1;
static const int dimensions[5] = {41024, 512, 32, 32, 1};
static const unsigned int NNUEVersion = 0x7AF32F16u;

extern void init_nnue(const char* path);
extern void load_nnue(const char* path, NNUE* nn);
extern void free_local_nnue(void);
extern void free_nnue(NNUE* nn);
extern void input_layer(const NNUE* nn, const board_t* b, const int color, int16_t* inp);
extern int evaluate_nnue(const board_t* b, const int useAcc);

extern void determine_changes(const int m, const board_t* b, NNUEChangeList* list);

extern void init_nnue_acc(const board_t* b);
extern void update_do(NNUEChangeList* q, const int m, const board_t* const b);
extern void update_undo(NNUEChangeList* q, const board_t* const b);

extern void perft_nnue(board_t b, int depth);

#endif