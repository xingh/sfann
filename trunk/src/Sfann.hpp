//
//   ------------------------------------------------------------------
//      Sfann v0.1 : Simple and Fast Artificial Neural Networks
//   ------------------------------------------------------------------
//
//      Copyright (C) 2010 Stanislas Oger
//
//   ..................................................................
//
//      This file is part of Sfann
//
//      Sfann is free software; you can redistribute it and/or modify
//      it under the terms of the GNU General Public License as published by
//      the Free Software Foundation; either version 2 of the License, or
//      (at your option) any later version.
//
//      This program is distributed in the hope that it will be useful,
//      but WITHOUT ANY WARRANTY; without even the implied warranty of
//      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//      GNU General Public License for more details.
//
//      You should have received a copy of the GNU General Public License
//      along with this program; if not, write to the Free Software
//      Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
//
//   ..................................................................
//
//      Contact :
//                stanislas.oger@gmail.com
//   ..................................................................
// 


#ifndef __LIB_SANN__
#define __LIB_SANN__

#include <iostream>
#include <string>
#include <iterator>
#include <algorithm>
#include <vector>
#include <map>

// pour random
#include <cstdlib>
#include <ctime>

#include <boost/program_options.hpp>
#include "fann.h"
#include "SfannException.hpp"


using namespace std;
using namespace boost::program_options;


typedef struct net_carac {
    struct fann * net;
    
    float train_perfs;
    int train_num_ok;
    float train_mse;
    int train_num_data;

    float dev_perfs;
    int dev_num_ok;
    int dev_num_data;
    fann_type ** dev_out;
    int dev_num_output;

    float test_perfs;
    int test_num_ok;
    int test_num_data;
    fann_type ** test_out;
    int test_num_output;
} net_carac;


typedef struct train_dev_test_couple {
    struct fann_train_data * train;
    struct fann_train_data * dev;
    struct fann_train_data * test;
} train_dev_test_couple;

typedef struct folds {
    struct fann_train_data * data;
    int num_folds;
    int num_data;
} folds;

typedef struct training_res {
    net_carac * net_max_train;
    net_carac * net_max_dev;
    net_carac * net_max_test;
} training_res;


class Sfann {

    private:
        Sfann();
        ~Sfann();

        options_description * options;
        variables_map * config;
        struct fann_train_data *train_data, *dev_data, *test_data;

        training_res * train_courant;

        static int max_struct(fann_type* output, int number);
        static void print_map(map<int, int> & m);
        static int max_struct(map<int, int> & output);
        static int perfs_on_data(struct fann * net, struct fann_train_data *data);
        static void perfs_on_data(struct fann * net, struct fann_train_data * data, int & nb_bons, fann_type ** & res);
        static void print_net_carac(net_carac * nc);
        static void print_training_res(training_res * t);
        // opérateurs sécurisés
        template <class T> static float divide_values(T a, T b);
        template <class T> static T multiply_values(T a, T b);
        template <class T> static T add_values(T a, T b);
        static void add_net_carac(net_carac * src, net_carac * & dest);
        static void add_training_res(training_res * src, training_res * dest);
        static void delete_fann_train_data(struct fann_train_data * & d, int nb);

        // callback FANN
        static int FANN_API training_callback(struct fann *ann, struct fann_train_data *train,unsigned int max_epochs, unsigned int epochs_between_reports,float desired_error, unsigned int epochs);
        // clone deux FANN
        template <class T> static T ** copy_matrix(T ** m, int x, int y);
        template <class T> static void delete_matrix(T ** & m, int x, int y);
        template <class T> static T** add_matrix_x(T ** m, int x, T ** m2, int x2, int y);
        static struct fann* fann_copy(const struct fann* orig);
        // alloue un <struct fann_train_data> pour accueillir num_data donnees, sur le modele de <src>, place le resultat dans <dest>
        static void init_structure_metadata(struct fann_train_data * src, struct fann_train_data * dest, int num_data);
        // ajoute toutes les donnes de <src> a <dest>
        static void copy_train_data(struct fann_train_data * src, struct fann_train_data * dest);
        // ajoute les <nb> donnes de <src> a partir de <start> vers <dest>
        static void copy_train_data(struct fann_train_data * src, struct fann_train_data * dest, int start, int nb);
        // genere le couple train/dev <cross_corpus> en prenant le fold <num_fold> comme dev et le reste comme train
        static void generate_cross_corpus(folds & _folds, train_dev_test_couple * cross_corpus, int num_test_fold, int nb_dev_folds);

        // lance la boucle d'apprentissage norale
        training_res * do_normal_training(int detail);

        // coupe le corpus de train en cross_nb_folds parties (folds) et met le resultat dans _folds
        static void generate_folds_from_train_corpus(struct fann_train_data * train_data, folds & _folds, int cross_nb_folds);
        // libere un tableau de train_dev_couple
        static void delete_train_dev_test_couple(train_dev_test_couple * & cross_corpora, int nb_corpora);
        // libere un folds
        static  void delete_folds(folds * & f);

        static void delete_training_res(training_res * & t, int nb);

        static void delete_net_carac(net_carac * & nc, int nb_nc);

        static net_carac * create_empty_net_carac();
        static training_res * create_training_res();

        static void create_dev_from_train_corpus(struct fann_train_data * & _dev_data, struct fann_train_data * & _train_data, const struct fann_train_data * _test_data, int dev_size) throw (SfannException);


// 		static net_carac * best_dev;
// 		static net_carac * best_train;

        static Sfann * me;


    public:
        static Sfann * getInstance();
        static void deleteInstance();

        void parse_config(int ac, char ** av) throw (exception);
        void check_options() throw (SfannException);
        // lance l'apprentissage (main)
        void do_training();
        void load_data();
        void usage();
};





#endif

