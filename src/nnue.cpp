#include <stdio.h>
#include <stdlib.h>
#include <string.h> //memcpy

#include "board.h"
#include "move.h"
#include "move_gen.h"
#include "move_do.h"
#include "nnue.h"
#include "nnuearch.h"


void read_headers(FILE* f);
void read_params(FILE* f, NNUE* nn);
void read_weights(FILE* f, weight_t* nn, const int dims, const int isOutput);

const u_int32_t FTHeader = 0x5d69d7b8;
const u_int32_t NTHeader = 0x63337156;
const u_int32_t NNUEHash = 0x3e5aa6eeU;
const int ArchSize = 177;

enum {
  PS_W_PAWN   =  1,
  PS_B_PAWN   =  1 * 64 + 1,
  PS_W_KNIGHT =  2 * 64 + 1,
  PS_B_KNIGHT =  3 * 64 + 1,
  PS_W_BISHOP =  4 * 64 + 1,
  PS_B_BISHOP =  5 * 64 + 1,
  PS_W_ROOK   =  6 * 64 + 1,
  PS_B_ROOK   =  7 * 64 + 1,
  PS_W_QUEEN  =  8 * 64 + 1,
  PS_B_QUEEN  =  9 * 64 + 1,
  PS_END     = 10 * 64 + 1
};

const u_int32_t PieceToIndex[2][16] = {
  { 0, PS_B_PAWN, PS_B_KNIGHT, PS_B_BISHOP, PS_B_ROOK, PS_B_QUEEN, 0, 0,
   0, PS_W_PAWN, PS_W_KNIGHT, PS_W_BISHOP, PS_W_ROOK, PS_W_QUEEN, 0, 0 },
  { 0, PS_W_PAWN, PS_W_KNIGHT, PS_W_BISHOP, PS_W_ROOK, PS_W_QUEEN, 0, 0,
   0, PS_B_PAWN, PS_B_KNIGHT, PS_B_BISHOP, PS_B_ROOK, PS_B_QUEEN, 0, 0 }
};


static NNUE nnue;
static int16_t nInput[kDimensionFT];

// functions

// CHECK_MALLOC()

static void CHECK_MALLOC(void* ptr) {
   if (!ptr) {
      fprintf(stderr, "Malloc failed in %s %d\n", __FILE__, __LINE__);
      exit(66);
   }
}

// CHECK_READ()

static void CHECK_READ(int read, int correct) {
   if (read != correct) {
      fprintf(stderr, "Unsuccessful read in %s %d\n", __FILE__, __LINE__);
      exit(5);
   }
}

// init_nnue()

void init_nnue(const char* path) {
   load_nnue(path, &nnue);
}

void load_nnue(const char* path, NNUE* nn) {
   ASSERT(sizeof(uint32_t) == 4);
   ASSERT(dimensions[2] == dimensions[3]);
   ASSERT(dimensions[2] == kDimensionHidden);
   ASSERT(kHalfDimensionFT == dimensions[1] / 2);
   ASSERT(kHalfDimensionFT == kDimensionFT / 2);

   FILE* f = fopen(path, "r");

   if (!f)
   {
      fprintf(stderr, "Can't open nnue file: %s\n", path);
      exit(5);
   }

   #ifdef NNUE_DEBUG
      printf("Loading NNUE %s\n", path);
   #endif

   nn->ftBiases = (int16_t*)malloc(sizeof(int16_t)*kHalfDimensionFT);
   nn->ftWeights = (int16_t*)malloc(sizeof(int16_t)*kHalfDimensionFT*kInputDimensionsFT);
   CHECK_MALLOC(nn->ftBiases);
   CHECK_MALLOC(nn->ftWeights);

   read_headers(f);
   read_params(f, nn);

   fclose(f);

   #ifdef NNUE_DEBUG
   if (0)
      show_nnue(nn);
   #endif

   printf("%s NNUE loaded\n", path);
}

void read_headers(FILE* f) {
   int32_t version, hash;
   int successfulRead = 1, size;

   successfulRead &= fread(&version, sizeof(u_int32_t), 1, f);
   successfulRead &= fread(&hash, sizeof(u_int32_t), 1, f);
   successfulRead &= fread(&size, sizeof(u_int32_t), 1, f);

   ASSERT(version == NNUEVersion);
   ASSERT(hash == NNUEHash);
   ASSERT(size == ArchSize);
   CHECK_READ(successfulRead, 1);

   char* architecture = (char*)malloc(ArchSize);
   CHECK_MALLOC(architecture);

   successfulRead = fread(architecture, sizeof(char), ArchSize, f);
   CHECK_READ(successfulRead, ArchSize);

   #ifdef NNUE_DEBUG
      printf("Version: %u\n", version);
      printf("Hash: %u\n", hash);
      printf("Size: %u\n", size);
      printf("Architecture: %s\n", architecture);
   #endif

   free(architecture);
}

//Code copied from evaluate_nnue
void read_params(FILE* f, NNUE* nn) {
   uint32_t header;
   int successfulRead = 1;

   //First read the feature transformer

   successfulRead = fread(&header, sizeof(u_int32_t), 1, f);
   CHECK_READ(successfulRead, 1);
   ASSERT(header == FTHeader);

   #ifdef NNUE_DEBUG
      printf("Header_FT: %u\n", header);
   #endif

   successfulRead = fread(nn->ftBiases, sizeof(nn->ftBiases[0]), kHalfDimensionFT, f);
   CHECK_READ(successfulRead, kHalfDimensionFT);

   successfulRead = fread(nn->ftWeights, sizeof(nn->ftWeights[0]), kHalfDimensionFT*kInputDimensionsFT, f);
   CHECK_READ(successfulRead, kHalfDimensionFT*kInputDimensionsFT);

   //Now read the network

   successfulRead = fread(&header, sizeof(u_int32_t), 1, f);
   CHECK_READ(successfulRead, 1);
   ASSERT(header == NTHeader);

   #ifdef NNUE_DEBUG
      printf("Header_Network: %u\n", header);
   #endif

   successfulRead = fread(nn->biases1, sizeof(nn->biases1[0]), dimensions[2], f);
   CHECK_READ(successfulRead, dimensions[3]);
   read_weights(f, nn->weights1, dimensions[1], 0);

   successfulRead = fread(nn->biases2, sizeof(nn->biases2[0]), dimensions[3], f);
   CHECK_READ(successfulRead, dimensions[3]);
   read_weights(f, nn->weights2, dimensions[2], 0);

   successfulRead = fread(nn->outputB, sizeof(nn->outputB[0]), dimensions[4], f);
   CHECK_READ(successfulRead, dimensions[4]);
   read_weights(f, nn->outputW, 1, 1);

   int ignore, cnt = 0;
   while (fread(&ignore, sizeof(int), 1, f))
      cnt++;
   if (cnt) {
      fprintf(stderr, "NNUE file hasn't been read completely, %d ints remeain\n", cnt);
      exit(10);
   }
}

void read_weights(FILE* f, weight_t* ws, const int dims, const int isOutput) {
   for (int i = 0; i < 32; ++i) {
      for (int j = 0; j < dims; ++j) {
         int8_t a;
         int s = fread(&a, 1, 1, f);
         CHECK_READ(s, 1);
         //Output is the same whether it is sparse or regular
         int idx = isOutput? i*dims+j : get_idx(i,j,dims);
         ws[idx] = (weight_t)a;
      }
   }
}

/*
//TODO: make this a "save nn into binary file" function
void show_nnue(const NNUE* nn)
{
   //./noc | tail --lines=+9 | sha256sum
   //./cfish | head --lines=-2 | tail --lines=+4 | sha256sum

   //Input will be needed
   //FT
   for (int i = 0; i < kHalfDimensionFT; ++i)
   {
      printf("%d, ", nn->ftBiases[i]);
   }
   printf("\n");

   for (int i = 0; i < kHalfDimensionFT*kInputDimensionsFT; ++i)
   {
      printf("%d, ", nn->ftWeights[i]);
   }
   printf("\n");

   //First layer
   for (int i = 0; i < 32; ++i)
   {
      printf("%d, ", nn->biases1[i]);
   }
   printf("\n");
   for (int i = 0; i < kDimensionFT*32; ++i)
   {
      printf("%d, ", nn->weights1[i]);
   }
   printf("\n");
   
   //Second layer
   for (int i = 0; i < 32; ++i)
   {
      printf("%d, ", nn->biases2[i]);
   }
   printf("\n");
   for (int i = 0; i < 32*32; ++i)
   {
      printf("%d, ", nn->weights2[i]);
   }
   printf("\n");

   //Output layer
   printf("%d\n", nn->outputB[0]);
   for (int i = 0; i < 1*32; ++i)
   {
      printf("%d, ", nn->outputW[i]);
   }
   printf("\n");
}
*/
void free_local_nnue(void) {
   free_nnue(&nnue);
}
void free_nnue(NNUE* nn) {
   free(nn->ftBiases);
   free(nn->ftWeights);
}

static int to_piece(int piece, const int colour) {
   ASSERT(piece > 0 && piece != NN_KING && !PIECE_IS_PAWN(piece));
   static int flags[6] = {0, QueenFlags, RookFlag, BishopFlag, KnightFlag};
   
   piece &= ~3;

   int a = 0, i;
   for (i = 1; i < 5; ++i)
   {
      if (piece == flags[i]) {
         a = 1;
         break;
      }
   }

   ASSERT(a==1);

   return (colour==White? 6 : 14) - i;
}

// to_piece_pawn()
static int to_piece_pawn(const int colour) {
   return (colour==White? 6 : 14) - 5; 
}

// to_sq()
static int to_sq(const int sq64, const int colour) {
   return sq64 ^ (colour==White? 0 : 0x3f);
}

// make_index()
static int make_index(const int colour, const int sq, const int piece, const int ksq) {
   return sq + PieceToIndex[1^colour][piece] + PS_END * ksq;
}

// input_layer()

void input_layer(const NNUE* nn, const board_t* const b, const int colour, int16_t* inp) {
//Calculates the input layer for a given colour (king-piece, king is of colour)
   //ASSERT(POPCOUNT(b->allPieces) <= 32);
   int actives[30];
   int numActives = 0;

   memcpy(inp, nn->ftBiases, sizeof(int16_t)*kHalfDimensionFT);

   const int ksq = to_sq(SQUARE_TO_64(KING_POS(b, colour)), colour);

   //GNU stuff
   const sq_t* ptr;
   int from, piece, sq, idx;
   int good_piece;

   const int king_mask = 0x80;

   for (int c = White; c <= Black; ++c) {
      for (ptr = &b->piece[c][0]; (from=*ptr) != SquareNone; ptr++) {
         piece = b->square[from];
         if (piece & KingFlag) //King is ignored
            continue;
         sq = to_sq(SQUARE_TO_64(from), colour);
         good_piece = to_piece(piece, c);
         idx = make_index(colour, sq, good_piece, ksq);
         actives[numActives++] = idx;
      }
      good_piece = to_piece_pawn(c);
      for (ptr = &b->pawn[c][0]; (from=*ptr) != SquareNone; ptr++) {
         sq = to_sq(SQUARE_TO_64(from), colour);
         piece = b->square[from];
         idx = make_index(colour, sq, good_piece, ksq);
         actives[numActives++] = idx;
      }
   }

   int16_t* ws = (int16_t*)nn->ftWeights;

   for (int i = 0; i < numActives; ++i) {
      const int offset = kHalfDimensionFT * actives[i];
      for (int j = 0; j < kHalfDimensionFT; ++j)
         inp[j] += ws[offset+j];
   }
}

// nnue_changes()

void nnue_change(NNUEChange* change, int piece, int sqr, int colour, int appears) {
   ASSERT(piece < 128);
   change->piece = piece;
   change->sqr = sqr;
   change->colour = colour;
   change->appears = appears;
}

// determine_changes()

void determine_changes(const int m, const board_t* b, NNUEChangeList* list)
{
#ifndef USE_NNUE
   return;
#endif

   const int colour = b->turn;
   const int from = MOVE_FROM(m);
   const int to = MOVE_TO(m);
   const int piece = b->square[from];
   const int capture = b->square[to];
   const int is_capture = capture != Empty;

   //If a KING moves, we have to reset everything
   if (PIECE_IS_KING(b->square[from])) {
      list->changes[0].piece = NN_KING;
      list->changes[0].sqr = m;
      list->changes[0].appears = 1;
      list->idx = 1;
      return;
   }

   ASSERT(piece && piece < 128);
   NNUEChange change;

   //Removing the piece from the current sqr
   nnue_change(&change, piece, from, colour, 0);
   list->changes[list->idx++] = change;
   //Place it where it goes to
   const int newPiece = MOVE_IS_PROMOTE(m)? move_promote(m) : piece;
   nnue_change(&change, newPiece, to, colour, 1);
   list->changes[list->idx++] = change;

   //En passand
   if (MOVE_IS_EN_PASSANT(m)) {
      nnue_change(&change, piece, SQUARE_EP_DUAL(to), 1^colour, 0);
      list->changes[list->idx++] = change;
   } else if (is_capture) {
   //If we captured a piece, remove it
      nnue_change(&change, capture, to, 1^colour, 0);
      list->changes[list->idx++] = change;
   }

   ASSERT(list->idx > 1);
}

// apply_changes()

void apply_changes(const NNUE* nn, const board_t* b, const NNUEChangeList* list, const int colour, int16_t* inp) {

   if (list->changes[0].piece == NN_KING) {
      input_layer(nn, b, colour, inp);
      return;
   }

   const int ksq = to_sq(SQUARE_TO_64(KING_POS(b, colour)), colour);

   for (int i = 0; i < list->idx; ++i) {
      const int c = list->changes[i].colour;
      const int piece = list->changes[i].piece;
      ASSERT(piece != NN_KING && piece < 128);
      const int good_piece = PIECE_IS_PAWN(piece)? to_piece_pawn(c) : to_piece(piece, c);
      const int sq = to_sq(SQUARE_TO_64(list->changes[i].sqr), colour);
      const int offset = kHalfDimensionFT * make_index(colour, sq, good_piece, ksq);

      int16_t* ws = nn->ftWeights;

      if (list->changes[i].appears) {
         for (int j = 0; j < kHalfDimensionFT; ++j)
            inp[j] += ws[offset+j];
      } else {
         for (int j = 0; j < kHalfDimensionFT; ++j)
            inp[j] -= ws[offset+j];
      }
   }
}

// init_nnue_acc()

void init_nnue_acc(const board_t* b) {
   input_layer(&nnue, b, White, nInput + kHalfDimensionFT);
   input_layer(&nnue, b, Black, nInput);
}

// update_do()

void update_do(NNUEChangeList* q, const int m, const board_t* b) {
#ifndef USE_NNUE
   return;
#endif

   apply_changes(&nnue, b, q, White, nInput + kHalfDimensionFT);
   apply_changes(&nnue, b, q, Black, nInput);

   ASSERT(q->idx < 5);
}

// update_undo()

void update_undo(NNUEChangeList* q, const board_t* b) {
#ifndef USE_NNUE
   return;
#endif

   ASSERT(q->idx < 5);

   for (int i = 0; i < q->idx; ++i)
      q->changes[i].appears ^= 1;

   apply_changes(&nnue, b, q, White, nInput + kHalfDimensionFT);
   apply_changes(&nnue, b, q, Black, nInput);

   q->idx = 0;
}

// evaluate_nnue()

int evaluate_nnue(const board_t* const b, const int useAcc)
{
   int ev;
   if (useAcc)
      ev = evaluate_acc(&nnue, b, nInput);
   else
      ev = evaluate(&nnue, b, nInput);
   return ev;
}

