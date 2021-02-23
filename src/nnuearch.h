
// nnuearch.h

#ifndef NNUEARCH_H
#define NNUEARCH_H

extern const int get_idx(const int i, const int j, const int dim);

extern int evaluate_acc(const NNUE* nn, const board_t* b, const int16_t* nInput);
extern int evaluate(const NNUE* nn, const board_t* b, int16_t* nInput);

enum {
    SHIFT = 6,
};

// clip64()

inline static const clipped_t clip64(const int32_t v) {
    //Clip but >> 64
    if (v <= 0) return 0;
    if (v >= 127 << SHIFT) return 127;
    return v >> SHIFT;
}

// clip()

inline static const clipped_t clip(const int16_t v) {
    return v >= 127? 127 : (v <= 0? 0 : v);
}

#endif
