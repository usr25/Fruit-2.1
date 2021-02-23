
// main.cpp

// includes

#include <cstdio>
#include <cstdlib>

#include "attack.h"
#include "book.h"
#include "hash.h"
#include "move_do.h"
#include "option.h"
#include "pawn.h"
#include "piece.h"
#include "protocol.h"
#include "random.h"
#include "square.h"
#include "trans.h"
#include "util.h"
#include "value.h"
#include "vector.h"

#include "nnue.h"

//Edit this to avoid having to pass the path
const char* NNUE_PATH = "-";

// functions

// main()

int main(int argc, char * argv[]) {

   // init

   util_init();
   my_random_init(); // for opening book

   printf("Fruit 2.1 UCI by Fabien Letouzey\n");
   printf("NNUE fork by Jorge Fernandez\n");

   // early initialisation (the rest is done after UCI options are parsed in protocol.cpp)

   option_init();

   square_init();
   piece_init();
   pawn_init_bit();
   value_init();
   vector_init();
   attack_init();
   move_do_init();

   random_init();
   hash_init();

   trans_init(Trans);
   book_init();

   if (NNUE_PATH[0] != '-')
      init_nnue(NNUE_PATH);
   else
      init_nnue(argv[1]);

   // loop

   loop();

   free_local_nnue();

   return EXIT_SUCCESS;
}

// end of main.cpp

