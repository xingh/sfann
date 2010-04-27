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


#include "Sfann.hpp"

Sfann * Sfann::me = NULL;
// net_carac * Sfann::best_dev = NULL;
// net_carac * Sfann::best_train = NULL;

Sfann::Sfann() {
    this->config = NULL;

    options_description generic("Generic options");
    generic.add_options()
        ("help,h", "prints this help message")
        ("verbose,v", "verbose outputs")
        ;

    options_description actions("Action to be performed");
    actions.add_options()
        ("do-cross-validation", "Perform a cross-validation on the train corpus")
        ("do-training", "train an ANN using the specified train/dev/test corpora")
        ("do-running", "run the specified ANN on the specified test corpus")
        ("do-nothing", "does not train ANN, only load data and generate corpora")
        ;

    options_description topology("ANN Specifications");
    topology.add_options()
// 		("num-input", value<int>(), "Size of the input layer")
// 		("num-output", value<int>(), "Size of the output layer")
        ("num-hidden,l", value<int>(), "Size of the hidden layer")
/*        ("write-max-dev", value<string>(), "Write in the specified file, the MLP that obtain the best score on the dev corpus")
        ("write-max-train", value<string>(), "Write in the specified file, the MLP that obtain the best score on the train corpus")
        ("write-max-test", value<string>(), "Write in the specified file, the MLP that obtain the best score on the test corpus")*/
        ;

    options_description data("Data-related options");
    data.add_options()
        ("train,t", value<string>(), "data file containing the training documents (fann data format)")
        ("dev,d", value<string>(), "data file containing the development documents (fann data format)")
        ("test,s", value<string>(), "data file containing the test documents (fann data format)")
        ("auto-dev,a", value<int>(), "automatically construct a dev corpus with <arg>% of the train")
        ("save-dev", value<string>(), "save the automatically build development corpus")
        ;

    options_description training("General training options");
    training.add_options()
        ("randomize-data,r", "Randomize the order of the elements in the data vectors before training")
        ("clever-init,i", "Use the Widrow and Nguyen's algorithm for initialize weights")
        ("reports", value<int>()->default_value(100), "Number of epoch between reports")
        ("max-epoch", value<int>()->default_value(5000), "Max epoch")
        ("desired-error", value<float>()->default_value(0.001), "Desired error")
        ("num-runs,n", value<int>()->default_value(1), "Number of training/testing cycles for finding the ANN that perform the best on the Dev")
//         ("-best-on", value<string>()->default_value("dev"), "The best ANN is the one that obtain the best results on the <arg> corpus, with <arg>=(dev|train|test)")
        ;

    options_description cv_opts("Do-cross-validation specific options");
    cv_opts.add_options()
        ("cv-num-folds", value<int>()->default_value(10), "specify the number of folds (k-folds) for cross validating on the train corpus")
        ("cv-num-dev", value<int>()->default_value(1), "specify the number of folds used for the dev corpus")
        ("cv-shuffle", "shuffle data before creating folds")
        ("leave-one-out,o", "use leave-one-out validation on the train corpus")
        ;
        
    options_description train_opts("Do-training specific options");
    train_opts.add_options()
        ("save-max-dev", value<string>(), "save the ANN that obtains the best score on the dev")
        ("save-max-train", value<string>(), "save the ANN that obtains the best score on the train")
        ("save-max-test", value<string>(), "save the ANN that obtains the best score on the test")
        ("save-max-test-run", value<string>(), "save the results on the test corpus of the best test ANN")
        ("save-max-dev-run", value<string>(), "save the results on the test corpus of the best dev ANN")
        ("save-max-train-run", value<string>(), "save the results on the test corpus of the best train ANN")
        ;
        
    options_description run_opts("Do-running specific options");
    run_opts.add_options()
        ("load-ann", value<string>(), "load the specified ANN")
        ("save-loaded-run", value<string>(), "save the results on the test corpus of the ANN loaded with --load-ann")
        ;

    this->options = new options_description();
    this->options->add(generic).add(actions).add(data).add(topology).add(training).add(cv_opts).add(train_opts).add(run_opts);

    this->train_courant = NULL;

    this->dev_data = NULL;
    this->test_data = NULL;
    this->train_data = NULL;
}


Sfann::~Sfann() {
    delete this->options;
    delete this->config;
    if (this->train_data != NULL) fann_destroy_train(this->train_data);
    if (this->test_data != NULL) fann_destroy_train(this->test_data);
    if (this->dev_data != NULL) fann_destroy_train(this->dev_data);
}


Sfann * Sfann::getInstance() {
    if (Sfann::me == NULL) {
        Sfann::me = new Sfann();
    }
    return Sfann::me;
}


void Sfann::deleteInstance() {
    if (Sfann::me != NULL) {
        delete Sfann::me;
        Sfann::me = NULL;
    }
}


void Sfann::parse_config(int argc, char ** argv) throw (exception) {
    this->config = new variables_map();
    store(parse_command_line(argc, argv, *this->options), *this->config);
    notify(*this->config);
}

void Sfann::check_options() throw(SfannException) {

    bool nothing = this->config->count("do-nothing");
    bool cross_validate = this->config->count("do-cross-validation");
    bool training = this->config->count("do-training");
    bool running = this->config->count("do-running");
    
    if (this->config->count("help")) {
        throw *new SfannException("Help required");
    }

    if (this->config->count("dev") && this->config->count("auto-dev")) {
        throw *new SfannException("Incompatible options : --dev and --auto-dev");
    }

    if ((training || cross_validate) && !this->config->count("train")) {
        throw *new SfannException("You have to specify a training corpus (with --train)");
    }

    if (this->config->count("do-running") + this->config->count("do-nothing") + this->config->count("do-training") + this->config->count("do-cross-validation") != 1) {
        throw *new SfannException("You have to specify (only) one action to be performed");
    }

    if (running && (!this->config->count("load-ann") || !this->config->count("test"))) {
        throw *new SfannException("You have to specify an ANN (with --load-ann) to run and a test corpus (with --test)");
    }

    if (training && (!this->config->count("num-hidden") || !this->config->count("num-runs") || !this->config->count("max-epoch") || !this->config->count("reports") || !this->config->count("desired-error"))) {
        throw *new SfannException("You have to specify more options for training ANN (--num-hidden missing ?)");
    }
}

void Sfann::usage() {
    cout << "Usage : sfann <action> [options]" << endl << *this->options << endl;
}


void Sfann::load_data() {
    if ((*this->config).count("train")) {
        string train = (*this->config)["train"].as<string>();

        cout << " ->  Reading " << train << " ...";
        this->train_data = fann_read_train_from_file(train.c_str());
        cout << " Ok ! (" << this->train_data->num_data << " examples)" << endl;
    }

    if ((*this->config).count("test")) {
        string test = (*this->config)["test"].as<string>();

        cout << " ->  Reading " << test << " ...";
        this->test_data = fann_read_train_from_file(test.c_str());
        cout << " Ok ! (" << this->test_data->num_data << " examples)" << endl;
    }

    if ((*this->config).count("dev")) {
        string dev = (*this->config)["dev"].as<string>();

        cout << " ->  Reading " << dev << " ...";
        this->dev_data = fann_read_train_from_file(dev.c_str());
        cout << " Ok ! (" << this->dev_data->num_data << " examples)" << endl;
    }

    if ((*this->config).count("auto-dev")) {
        delete this->dev_data; this->dev_data = NULL;
        create_dev_from_train_corpus(this->dev_data, this->train_data, this->test_data, (*this->config)["auto-dev"].as<int>());
    }

    if ((*this->config).count("save-dev")) {
        fann_save_train(this->dev_data, (*this->config)["save-dev"].as<string>().c_str());
    }
}

void Sfann::print_map(map<int, int> & m) {
    int total = 0;
    for (map<int,int>::iterator it = m.begin(); it != m.end(); ++it) {
        cout << "  - " << it->first << " -> " << it->second << "\n";
        total += it->second;
    }
}

// Extrait un corpus de dev des données de train en choisissant aleatoirement les exemples mais en en conservant la meme proportion que dans le test ; si _test_data est NULL alors la proportion du train sera gardee.
void Sfann::create_dev_from_train_corpus(struct fann_train_data * & _dev_data, struct fann_train_data * & _train_data, const struct fann_train_data * _test_data, int dev_size) throw (SfannException) {
    if (dev_size > 50) dev_size = 50;
    if (dev_size < 0) dev_size = 0;

    if (_train_data == NULL) {
        return;
    }

    // parcours le test pour connaitre sa composition (proportion de chaque classe)
    map<int, int> composition_test;
    int taille_test = 0;
    if (_test_data != NULL) {
        taille_test = _test_data->num_data;
        for (int i=0; i<taille_test; i++) {
            ++composition_test[max_struct(_test_data->output[i], _test_data->num_output)];
        }
    }

    // parcours le train pour connaitre sa composition (proportion de chaque classe)
    // pour verifier qu'il y a bien au moins le nombre d'elements qu'on veut pour le dev
    map<int, int> composition_train;
    int taille_train = _train_data->num_data;
    for (int i=0; i<taille_train; i++) {
        ++composition_train[max_struct(_train_data->output[i], _train_data->num_output)];
    }

    int taille_dev = _train_data->num_data * dev_size / 100;
    if (taille_dev == 0) taille_dev = 1;

    // calcul de la composition du dev
    map<int, int> composition_dev;
    int taille_dev_actuelle = 0;
    if (!composition_test.empty()) {
        for (map<int,int>::iterator it=composition_test.begin(); it != composition_test.end(); ++it) {
            composition_dev[it->first] = floor(((float)it->second/taille_test)*taille_dev);
            // il faut au moins autant d'exemples d'une classe dans le train que dans le dev
            if (composition_dev[it->first] > floor((float)composition_train[it->first]/2)) {
                composition_dev[it->first] = floor((float)composition_train[it->first]/2);
            }
            taille_dev_actuelle += composition_dev[it->first];
        }
    } else {
        for (map<int,int>::iterator it=composition_train.begin(); it != composition_train.end(); ++it) {
            composition_dev[it->first] = round(((float)it->second/taille_train)*taille_dev);
            // il faut au moins autant d'exemples d'une classe dans le train que dans le dev
            if (composition_dev[it->first] > floor((float)composition_train[it->first]/2)) {
                composition_dev[it->first] = floor((float)composition_train[it->first]/2);
            }
            taille_dev_actuelle += composition_dev[it->first];
        }
    }

    // ajout des elements manquants en piochant dans la classe dominante
    int max_classe = max_struct(composition_train);
    composition_dev[max_classe] += taille_dev - taille_dev_actuelle;

    cout << "Building dev corpus with " << taille_dev << " randomly choosen exemples in the " << _train_data->num_data << " exemples from the train corpus (" << dev_size << "%).\n";

    cout << "Train content :\n";
    print_map(composition_train);

    cout << "Test content :\n";
    print_map(composition_test);

    cout << "Dev content (" << dev_size << "%) :\n";
    print_map(composition_dev);

    // On alloue l'espace pour mettre le train moins le dev (le nouveau train)
    struct fann_train_data * new_train_data = new struct fann_train_data;
    init_structure_metadata(_train_data, new_train_data, taille_train-taille_dev);

    // On alloue l'espace pour mettre le dev
    _dev_data = new struct fann_train_data;
    init_structure_metadata(_train_data, _dev_data, taille_dev);

    // On choisit aleatoirement des indices du train, on ajoute l'exemple dans le dev si il est pas deja
    // ajoute, et si on a encore besoin d'exemple de cette classe
    set<int> deja;
    srand((unsigned)time(0));
    while (_dev_data->num_data < taille_dev) {
        int i = (rand()/(double)RAND_MAX)*(taille_train-1);
        // si on a pas deja vu cet exemple,
        if (deja.count(i) == 0) {
            int classe = max_struct(_train_data->output[i], _train_data->num_output);
            // et si on a encore besoin d'un exemple de cette classe
            if (composition_dev[classe] > 0) {
                // on copie l'exemple i dans le dev
                copy_train_data(_train_data, _dev_data, i, 1);
                deja.insert(i);
                composition_dev[classe]--;
            }
        }
    }

    // On remplit le nouveau train avec ce qui n'a pas ete mis dans le dev
    int pos = 0;
    int total = 0;
    for (set<int>::iterator it = deja.begin(); it != deja.end(); ++it) {
        if ((*it) != pos) {
            // cerr << (*it) << " -> copie " << (*it)-pos << " a partir de " << pos << endl;
            copy_train_data(_train_data, new_train_data, pos, (*it)-pos);
            total += (*it)-pos;
        }
        pos = (*it)+1;
    }
    if (pos < taille_train) {
        copy_train_data(_train_data, new_train_data, pos, taille_train-pos);
        total += taille_train-pos;
    }

    //cerr << total << " elements copies !" << endl;

    if (new_train_data->num_data != taille_train-taille_dev) {
        ostringstream oss;
        oss << "Le nouveau train fait " << new_train_data->num_data << " alors qu'il doit faire " << taille_train-taille_dev << ".";
        throw *new SfannException(oss.str());
    }

    fann_destroy_train(_train_data);
    _train_data = new_train_data;
}

int Sfann::max_struct(fann_type* output, int number) {
    int max = 0;
    for(int i = 0; i < number; i++) {
        if(output[max] < output[i]) {
            max = i;
        }
    }
    return max;
}

int Sfann::max_struct(map<int, int> & output) {
    map<int, int>::iterator it = output.begin();
    int max = it->first;
    for(; it != output.end(); ++it) {
        if(output[max] < it->second) {
            max = it->first;
        }
    }
    return max;
}


int FANN_API Sfann::training_callback(struct fann *ann, struct fann_train_data *train,unsigned int max_epochs, unsigned int epochs_between_reports,float desired_error, unsigned int epochs) {
    Sfann * me = Sfann::me;
    bool verbose = (*me->config).count("verbose");

    if (epochs == 1 && verbose) {
        ostringstream h;
        ostringstream l;
        char head[1024];
        char line[1024];
        sprintf(head, ": %-8s : %-10s : %-8s", "Epoch", "Train MSE", "Bit Fail");
        sprintf(line, "+----------+------------+----------");
        h << head;
        l << line;
        if (me->dev_data != NULL) {
            sprintf(head, " : %-8s : %-12s", "Dev MSE", "Dev c. rate");
            sprintf(line, "+----------+--------------");
            h << head;
            l << line;
        }
        if (me->test_data != NULL) {
            sprintf(head, " : %-8s : %-12s", "Tst MSE", "Tst c. rate");
            sprintf(line, "+----------+--------------");
            h << head;
            l << line;
        }
        cout << "\n";
        cout << h.str() << endl;
        cout << l.str() << endl;
    }

    float train_MSE = fann_get_MSE(ann);

    if (verbose) {
        unsigned int newBitFail = fann_get_bit_fail(ann);

        printf(": %08d : %.6f   : %8d", epochs, train_MSE , newBitFail);
    }

    int dev_num_ok = -1;
    float dev_perfs = -1;
    fann_type ** dev_out = NULL;
    if (me->dev_data != NULL && me->dev_data->num_data > 0) {
        me->perfs_on_data(ann, me->dev_data, dev_num_ok, dev_out);
        dev_perfs = (float)dev_num_ok/me->dev_data->num_data;

        if (verbose) {
            float dev_mse = fann_test_data(ann, me->dev_data);
            printf(" : %.6f : %10.2f %%", dev_mse, dev_perfs*100);
        }
    }

    int test_num_ok = -1;
    float test_perfs = -1;
    fann_type ** test_out = NULL;
    if (me->test_data != NULL && me->test_data->num_data > 0) {
        me->perfs_on_data(ann, me->test_data, test_num_ok, test_out);
        test_perfs = (float)test_num_ok/me->test_data->num_data;

        if (verbose) {
            float test_mse = fann_test_data(ann, me->test_data);
            printf(" : %.6f : %10.2f %%", test_mse, test_perfs*100);
        }
    }

    if (verbose) printf("\n");

    if (test_num_ok >= 0 && (me->train_courant->net_max_test == NULL || me->train_courant->net_max_test->test_perfs < test_perfs)) {
	// fprintf(stderr, "chk1\n");
        if (me->train_courant->net_max_test != NULL) delete_net_carac(me->train_courant->net_max_test, 1);
	// fprintf(stderr, "chk1\n");

        me->train_courant->net_max_test = create_empty_net_carac();
        // fprintf(stderr, "chk1\n");
        me->train_courant->net_max_test->net = me->fann_copy(ann);
        // fprintf(stderr, "chk1\n");

        if (me->train_data != NULL) {
            me->train_courant->net_max_test->train_mse = train_MSE;
            me->train_courant->net_max_test->train_num_data = me->train_data->num_data;
        }

        if (me->dev_data != NULL) {
            me->train_courant->net_max_test->dev_perfs = dev_perfs;
            me->train_courant->net_max_test->dev_num_ok = dev_num_ok;
            me->train_courant->net_max_test->dev_out = copy_matrix<fann_type>(dev_out, me->dev_data->num_data, me->dev_data->num_output);
            me->train_courant->net_max_test->dev_num_output = me->dev_data->num_output;
            me->train_courant->net_max_test->dev_num_data = me->dev_data->num_data;
        }

        if (me->test_data != NULL) {
            me->train_courant->net_max_test->test_perfs = test_perfs;
            me->train_courant->net_max_test->test_num_ok = test_num_ok;
            me->train_courant->net_max_test->test_out = copy_matrix<fann_type>(test_out, me->test_data->num_data, me->test_data->num_output);
            me->train_courant->net_max_test->test_num_output = me->test_data->num_output;
            me->train_courant->net_max_test->test_num_data = me->test_data->num_data;
        }
        // fprintf(stderr, "chk1\n");
    }


    if (dev_num_ok >= 0 && (me->train_courant->net_max_dev == NULL || me->train_courant->net_max_dev->dev_perfs < dev_perfs)) {
        // fprintf(stderr, "chk2\n");
        if (me->train_courant->net_max_dev != NULL) delete_net_carac(me->train_courant->net_max_dev, 1);
        // fprintf(stderr, "chk2\n");

        me->train_courant->net_max_dev = create_empty_net_carac();
        me->train_courant->net_max_dev->net = me->fann_copy(ann);

        if (me->train_data != NULL) {
            me->train_courant->net_max_dev->train_mse = train_MSE;
            me->train_courant->net_max_dev->train_num_data = me->train_data->num_data;
        }

        if (me->dev_data != NULL) {
            me->train_courant->net_max_dev->dev_perfs = dev_perfs;
            me->train_courant->net_max_dev->dev_num_ok = dev_num_ok;
            me->train_courant->net_max_dev->dev_out = copy_matrix<fann_type>(dev_out, me->dev_data->num_data, me->dev_data->num_output);
            me->train_courant->net_max_dev->dev_num_output = me->dev_data->num_output;
            me->train_courant->net_max_dev->dev_num_data = me->dev_data->num_data;
        }

        if (me->test_data != NULL) {
            me->train_courant->net_max_dev->test_perfs = test_perfs;
            me->train_courant->net_max_dev->test_num_ok = test_num_ok;
            me->train_courant->net_max_dev->test_out = copy_matrix<fann_type>(test_out, me->test_data->num_data, me->test_data->num_output);
            me->train_courant->net_max_dev->test_num_output = me->test_data->num_output;
            me->train_courant->net_max_dev->test_num_data = me->test_data->num_data;
        }
    }

    if (me->train_courant->net_max_train == NULL || me->train_courant->net_max_train->train_mse < 0 || me->train_courant->net_max_train->train_mse > train_MSE) {
        // fprintf(stderr, "chk3\n");
        if (me->train_courant->net_max_train != NULL) delete_net_carac(me->train_courant->net_max_train, 1);
        // fprintf(stderr, "chk3\n");

        me->train_courant->net_max_train = create_empty_net_carac();
        me->train_courant->net_max_train->net = me->fann_copy(ann);

        if (me->train_data != NULL) {
            me->train_courant->net_max_train->train_mse = train_MSE;
            me->train_courant->net_max_train->train_num_data = me->train_data->num_data;
        }

        if (me->dev_data != NULL) {
            me->train_courant->net_max_train->dev_perfs = dev_perfs;
            me->train_courant->net_max_train->dev_num_ok = dev_num_ok;
            me->train_courant->net_max_train->dev_out = copy_matrix<fann_type>(dev_out, me->dev_data->num_data, me->dev_data->num_output);
            me->train_courant->net_max_train->dev_num_output = me->dev_data->num_output;
            me->train_courant->net_max_train->dev_num_data = me->dev_data->num_data;
        }

        if (me->test_data != NULL) {
            me->train_courant->net_max_train->test_perfs = test_perfs;
            me->train_courant->net_max_train->test_num_ok = test_num_ok;
            me->train_courant->net_max_train->test_out = copy_matrix<fann_type>(test_out, me->test_data->num_data, me->test_data->num_output);
            me->train_courant->net_max_train->test_num_output = me->test_data->num_output;
            me->train_courant->net_max_train->test_num_data = me->test_data->num_data;
        }
    }

/*
    if (me->best_train->net == NULL || me->best_train->train_mse > trainMSE) {
        me->best_train->train_mse = trainMSE;
        if (me->best_train->net != NULL) {
            fann_destroy(me->best_train->net);
        }
        me->best_train->net = fann_copy(ann);
        cerr << "sauve train" << endl;
    }
*/


    return 1;
}

struct fann* Sfann::fann_copy(const struct fann* orig) {
/* deep copy of the fann structure */
/* adapted from FANN CVS */

    struct fann* copy;
    unsigned int num_layers = orig->last_layer - orig->first_layer;
    struct fann_layer *orig_layer_it, *copy_layer_it;
    unsigned int layer_size;
    struct fann_neuron *last_neuron,*orig_neuron_it,*copy_neuron_it;
    unsigned int i;
    struct fann_neuron *orig_first_neuron,*copy_first_neuron;
    unsigned int input_neuron;

    copy = fann_allocate_structure(num_layers);
    if (copy==NULL) {
        fann_error((struct fann_error*)orig, FANN_E_CANT_ALLOCATE_MEM);
        return NULL;
    }
    copy->errno_f = orig->errno_f;
    if (orig->errstr)
    {
        copy->errstr = (char *) malloc(FANN_ERRSTR_MAX);
        if (copy->errstr == NULL)
        {
            fann_destroy(copy);
            return NULL;
        }
        strcpy(copy->errstr,orig->errstr);
    }
    copy->error_log = orig->error_log;

    copy->learning_rate = orig->learning_rate;
    copy->learning_momentum = orig->learning_momentum;
    copy->connection_rate = orig->connection_rate;
    copy->network_type = orig->network_type;
    copy->num_MSE = orig->num_MSE;
    copy->MSE_value = orig->MSE_value;
    copy->num_bit_fail = orig->num_bit_fail;
    copy->bit_fail_limit = orig->bit_fail_limit;
    copy->train_error_function = orig->train_error_function;
    copy->train_stop_function = orig->train_stop_function;
    copy->callback = orig->callback;
    copy->cascade_output_change_fraction = orig->cascade_output_change_fraction;
    copy->cascade_output_stagnation_epochs = orig->cascade_output_stagnation_epochs;
    copy->cascade_candidate_change_fraction = orig->cascade_candidate_change_fraction;
    copy->cascade_candidate_stagnation_epochs = orig->cascade_candidate_stagnation_epochs;
    copy->cascade_best_candidate = orig->cascade_best_candidate;
    copy->cascade_candidate_limit = orig->cascade_candidate_limit;
    copy->cascade_weight_multiplier = orig->cascade_weight_multiplier;
    copy->cascade_max_out_epochs = orig->cascade_max_out_epochs;
    copy->cascade_max_cand_epochs = orig->cascade_max_cand_epochs;
    copy->user_data = orig->user_data;

/* copy cascade activation functions */
    copy->cascade_activation_functions_count = orig->cascade_activation_functions_count;
    copy->cascade_activation_functions = (enum fann_activationfunc_enum *)realloc(copy->cascade_activation_functions,
        copy->cascade_activation_functions_count * sizeof(enum fann_activationfunc_enum));
    if(copy->cascade_activation_functions == NULL)
    {
        fann_error((struct fann_error*)orig, FANN_E_CANT_ALLOCATE_MEM);
        fann_destroy(copy);
        return NULL;
    }
    memcpy(copy->cascade_activation_functions,orig->cascade_activation_functions,
            copy->cascade_activation_functions_count * sizeof(enum fann_activationfunc_enum));

    /* copy cascade activation steepnesses */
    copy->cascade_activation_steepnesses_count = orig->cascade_activation_steepnesses_count;
    copy->cascade_activation_steepnesses = (fann_type *)realloc(copy->cascade_activation_steepnesses, copy->cascade_activation_steepnesses_count * sizeof(fann_type));
    if(copy->cascade_activation_steepnesses == NULL)
    {
        fann_error((struct fann_error*)orig, FANN_E_CANT_ALLOCATE_MEM);
        fann_destroy(copy);
        return NULL;
    }
    memcpy(copy->cascade_activation_steepnesses,orig->cascade_activation_steepnesses,copy->cascade_activation_steepnesses_count * sizeof(fann_type));

    copy->cascade_num_candidate_groups = orig->cascade_num_candidate_groups;

    /* copy candidate scores, if used */
    if (orig->cascade_candidate_scores == NULL)
    {
        copy->cascade_candidate_scores = NULL;
    }
    else
    {
        copy->cascade_candidate_scores =
            (fann_type *) malloc(fann_get_cascade_num_candidates(copy) * sizeof(fann_type));
        if(copy->cascade_candidate_scores == NULL)
        {
            fann_error((struct fann_error *) orig, FANN_E_CANT_ALLOCATE_MEM);
            fann_destroy(copy);
            return NULL;
        }
        memcpy(copy->cascade_candidate_scores,orig->cascade_candidate_scores,fann_get_cascade_num_candidates(copy) * sizeof(fann_type));
    }

    copy->quickprop_decay = orig->quickprop_decay;
    copy->quickprop_mu = orig->quickprop_mu;
    copy->rprop_increase_factor = orig->rprop_increase_factor;
    copy->rprop_decrease_factor = orig->rprop_decrease_factor;
    copy->rprop_delta_min = orig->rprop_delta_min;
    copy->rprop_delta_max = orig->rprop_delta_max;
    copy->rprop_delta_zero = orig->rprop_delta_zero;

    /* user_data is not deep copied.  user should use fann_copy_with_user_data() for that */
    copy->user_data = orig->user_data;

#ifdef FIXEDFANN
    copy->decimal_point = orig->decimal_point;
    copy->multiplier = orig->multiplier;
    memcpy(copy->sigmoid_results,orig->sigmoid_results,6*sizeof(fann_type));
    memcpy(copy->sigmoid_values,orig->sigmoid_values,6*sizeof(fann_type));
    memcpy(copy->sigmoid_symmetric_results,orig->sigmoid_symmetric_results,6*sizeof(fann_type));
    memcpy(copy->sigmoid_symmetric_values,orig->sigmoid_symmetric_values,6*sizeof(fann_type));
#endif


    /* copy layer sizes, prepare for fann_allocate_neurons */
    for (orig_layer_it = orig->first_layer, copy_layer_it = copy->first_layer;
            orig_layer_it != orig->last_layer; orig_layer_it++, copy_layer_it++)
    {
        layer_size = orig_layer_it->last_neuron - orig_layer_it->first_neuron;
        copy_layer_it->first_neuron = NULL;
        copy_layer_it->last_neuron = copy_layer_it->first_neuron + layer_size;
        copy->total_neurons += layer_size;
    }
    copy->num_input = orig->num_input;
    copy->num_output = orig->num_output;


    /* copy scale parameters, when used */
#ifndef FIXEDFANN
    if (orig->scale_mean_in != NULL)
    {
        fann_allocate_scale(copy);
        for (i=0; i < orig->num_input ; i++) {
            copy->scale_mean_in[i] = orig->scale_mean_in[i];
            copy->scale_deviation_in[i] = orig->scale_deviation_in[i];
            copy->scale_new_min_in[i] = orig->scale_new_min_in[i];
            copy->scale_factor_in[i] = orig->scale_factor_in[i];
        }
        for (i=0; i < orig->num_output ; i++) {
            copy->scale_mean_out[i] = orig->scale_mean_out[i];
            copy->scale_deviation_out[i] = orig->scale_deviation_out[i];
            copy->scale_new_min_out[i] = orig->scale_new_min_out[i];
            copy->scale_factor_out[i] = orig->scale_factor_out[i];
        }
    }
#endif

    /* copy the neurons */
    fann_allocate_neurons(copy);
    if (copy->errno_f == FANN_E_CANT_ALLOCATE_MEM)
    {
        fann_destroy(copy);
        return NULL;
    }
    layer_size = (orig->last_layer-1)->last_neuron - (orig->last_layer-1)->first_neuron;
    memcpy(copy->output,orig->output, layer_size * sizeof(fann_type));

    last_neuron = (orig->last_layer - 1)->last_neuron;
    for (orig_neuron_it = orig->first_layer->first_neuron, copy_neuron_it = copy->first_layer->first_neuron;
            orig_neuron_it != last_neuron; orig_neuron_it++, copy_neuron_it++)
    {
        memcpy(copy_neuron_it,orig_neuron_it,sizeof(struct fann_neuron));
    }
/* copy the connections */
    copy->total_connections = orig->total_connections;
    fann_allocate_connections(copy);
    if (copy->errno_f == FANN_E_CANT_ALLOCATE_MEM)
    {
        fann_destroy(copy);
        return NULL;
    }

    orig_first_neuron = orig->first_layer->first_neuron;
    copy_first_neuron = copy->first_layer->first_neuron;
    for (i=0; i < orig->total_connections; i++)
    {
        copy->weights[i] = orig->weights[i];
        input_neuron = orig->connections[i] - orig_first_neuron;
        copy->connections[i] = copy_first_neuron + input_neuron;
    }

    if (orig->train_slopes)
    {
        copy->train_slopes = (fann_type *) malloc(copy->total_connections_allocated * sizeof(fann_type));
        if (copy->train_slopes == NULL)
        {
            fann_error((struct fann_error *) orig, FANN_E_CANT_ALLOCATE_MEM);
            fann_destroy(copy);
            return NULL;
        }
        memcpy(copy->train_slopes,orig->train_slopes,copy->total_connections_allocated * sizeof(fann_type));
    }

    if (orig->prev_steps)
    {
        copy->prev_steps = (fann_type *) malloc(copy->total_connections_allocated * sizeof(fann_type));
        if (copy->prev_steps == NULL)
        {
            fann_error((struct fann_error *) orig, FANN_E_CANT_ALLOCATE_MEM);
            fann_destroy(copy);
            return NULL;
        }
        memcpy(copy->prev_steps, orig->prev_steps, copy->total_connections_allocated * sizeof(fann_type));
    }

    if (orig->prev_train_slopes)
    {
        copy->prev_train_slopes = (fann_type *) malloc(copy->total_connections_allocated * sizeof(fann_type));
        if (copy->prev_train_slopes == NULL)
        {
            fann_error((struct fann_error *) orig, FANN_E_CANT_ALLOCATE_MEM);
            fann_destroy(copy);
            return NULL;
        }
        memcpy(copy->prev_train_slopes,orig->prev_train_slopes, copy->total_connections_allocated * sizeof(fann_type));
    }

    if (orig->prev_weights_deltas)
    {
        copy->prev_weights_deltas = (fann_type *) malloc(copy->total_connections_allocated * sizeof(fann_type));
        if(copy->prev_weights_deltas == NULL)
        {
            fann_error((struct fann_error *) orig, FANN_E_CANT_ALLOCATE_MEM);
            fann_destroy(copy);
            return NULL;
        }
        memcpy(copy->prev_weights_deltas, orig->prev_weights_deltas,copy->total_connections_allocated * sizeof(fann_type));
    }

    return copy;
}


void Sfann::perfs_on_data(struct fann * net, struct fann_train_data * data, int & nb_bons, fann_type ** & res) {
    nb_bons = 0;
    int num_data = data->num_data;
    int num_output = data->num_output;

    if (res == NULL) {
        res = new fann_type*[num_data];
        for (int i=0; i<num_data; i++) {
            res[i] = new fann_type[num_output];
        }
    }
    
    for (int i=0; i<num_data; i++) {
        res[i] = fann_run(net, data->input[i]);
        if( max_struct(data->output[i], data->num_output) == max_struct(res[i], data->num_output) ) {
            nb_bons++;
        }
    }
}

int Sfann::perfs_on_data(struct fann * net, struct fann_train_data * data) {
    int nb_bons = 0;
    int nb_dev_data = data->num_data;

    fann_type * out;
    for (int i=0; i<nb_dev_data; i++) {
        out = fann_run(net, data->input[i]);
        if( max_struct(data->output[i], data->num_output) == max_struct(out, data->num_output) ) {
            nb_bons++;
        }
    }

    return nb_bons;
}


training_res * Sfann::create_training_res() {
    training_res * t = new training_res[1];
    t->net_max_train = NULL;//create_empty_net_carac();
    t->net_max_dev = NULL;//create_empty_net_carac();
    t->net_max_test = NULL;//create_empty_net_carac();
    return t;
}

void Sfann::delete_training_res(training_res * & t, int nb_nc) {
    if (t == NULL) {return;}

    for (int k=0; k<nb_nc; k++) {
        if (t[k].net_max_train != NULL) delete_net_carac(t[k].net_max_train, 1);
        if (t[k].net_max_test != NULL) delete_net_carac(t[k].net_max_test, 1);
        if (t[k].net_max_dev != NULL) delete_net_carac(t[k].net_max_dev, 1);
    }
    delete[] t;
    t = NULL;
}

template <class T>
T Sfann::add_values(T a, T b) {
    if (a < 0 && b < 0) {
        return -1;
    } else {
        if (a < 0) a = 0;
        if (b < 0) b = 0;
        return a + b;
    }
}

template <class T>
float Sfann::divide_values(T a, T b) {
    if (a < 0 || b <= 0) {
        return -1;
    } else {
        return (float)a / b;
    }
}

template <class T>
T Sfann::multiply_values(T a, T b) {
    if (a <= 0 || b <= 0) {
        return 0;
    } else {
        return (T)a * b;
    }
}

net_carac * Sfann::create_empty_net_carac() {
    net_carac * nc = new net_carac[1];

    nc->net = NULL;

    nc->train_mse = -1;

    nc->train_num_ok = -1;
    nc->train_num_data = -1;

    nc->train_perfs = -1;

    nc->dev_num_ok = -1;
    nc->dev_num_data = -1;
    nc->dev_perfs = -1;
    nc->dev_out = NULL;

    nc->test_num_ok = -1;
    nc->test_num_data = -1;
    nc->test_perfs = -1;
    nc->test_out = NULL;

    return nc;
}

template <class T>
void Sfann::delete_matrix(T ** & m, int x, int y) {
    if (m == NULL) return;
    for (int i=0; i<x; i++) {
        delete[] m[i];
    }
    delete[] m;
    m = NULL;
}

void Sfann::add_net_carac(net_carac * src, net_carac * & dest) {
    if (src == NULL) return;
    if (dest == NULL) dest = create_empty_net_carac();

    float dest_mse = multiply_values<float>(dest->train_mse, dest->train_num_data);
    float src_mse = multiply_values<float>(src->train_mse, src->train_num_data);
    int total_data = add_values<int>(src->train_num_data, dest->train_num_data);

    dest->train_mse = divide_values<float>(add_values<float>(dest_mse, src_mse), total_data);

    dest->train_num_ok = add_values<int>(src->train_num_ok, dest->train_num_ok);
    dest->train_num_data = total_data;

    dest->train_perfs = divide_values<int>(dest->train_num_ok, dest->train_num_data);

    dest->dev_num_ok = add_values<int>(dest->dev_num_ok, src->dev_num_ok);
    dest->dev_num_data = add_values<int>(dest->dev_num_data, src->dev_num_data);

    if (dest->dev_out != NULL) delete_matrix<fann_type>(dest->dev_out, dest->dev_num_data, dest->dev_num_output);
    dest->dev_out = NULL; // add_matrix_x<fann_type>(dest->dev_out, dest->dev_num_data, src->dev_out, src->dev_num_data, src->dev_num_output);

    dest->dev_perfs = divide_values<int>(dest->dev_num_ok, dest->dev_num_data);

    dest->test_num_ok = add_values<int>(dest->test_num_ok, src->test_num_ok);
    dest->test_num_data = add_values<int>(dest->test_num_data, src->test_num_data);
    dest->test_perfs = divide_values<int>(dest->test_num_ok, dest->test_num_data);
    if (dest->test_out != NULL) delete_matrix<fann_type>(dest->test_out, dest->test_num_data, dest->test_num_output);
    dest->test_out = NULL; //add_matrix_x<fann_type>(dest->test_out, dest->test_num_data, src->test_out, src->test_num_data, src->test_num_output);
}

void Sfann::add_training_res(training_res * src, training_res * dest) {
    if (src == NULL || dest == NULL) return;
    add_net_carac(src->net_max_train, dest->net_max_train);
    add_net_carac(src->net_max_test, dest->net_max_test);
    add_net_carac(src->net_max_dev, dest->net_max_dev);
}


void Sfann::print_net_carac(net_carac * nc) {
    if (nc != NULL) {
        if (nc->train_mse >= 0) printf("train-mse=%.2f ", nc->train_mse);
        if (nc->dev_perfs >= 0) printf("dev=%.2f ",nc->dev_perfs*100);
        if (nc->test_perfs >= 0) printf("test=%.2f ",nc->test_perfs*100);
    }
}

void Sfann::print_training_res(training_res * t) {
        if (t == NULL) {
            throw * new SfannException("Error : trying to print null training_res !");
            return;
        }
        if (t->net_max_train != NULL) {
            printf("-> Classif max sur le train : ");
            print_net_carac(t->net_max_train);
            printf("\n");
        }
        if (t->net_max_dev != NULL) {
            printf("-> Classif max sur le dev   : ");
            print_net_carac(t->net_max_dev);
            printf("\n");
        }
        if (t->net_max_test != NULL) {
            printf("-> Classif max sur le test  : ");
            print_net_carac(t->net_max_test);
            printf("\n");
        }
}

void Sfann::delete_net_carac(net_carac * & nc, int nb_nc) {
    if (nc == NULL) return;

    for (int k=0; k<nb_nc; k++) {
        if (nc[k].net != NULL) {
            fann_destroy(nc[k].net);
        }
        if (nc[k].dev_out != NULL) {
            for (int i=0; i<nc[k].dev_num_data; i++) {
                delete[] nc[k].dev_out[i];
            }
            delete[] nc[k].dev_out;
        }
        if (nc[k].test_out != NULL) {
            for (int i=0; i<nc[k].test_num_data; i++) {
                delete[] nc[k].test_out[i];
            }
            delete[] nc[k].test_out;
        }
    }
    delete[] nc;
    nc = NULL;
}

template <class T>
T ** Sfann::copy_matrix(T ** m, int x, int y) {
    T ** m2 = new T*[x];
    
    for (int i=0; i<x; i++) {
        m2[i] = new T[y];
        for (int j=0; j<y; j++) {
            m2[i][j] = m[i][j];
        }
    }
    
    return m2;
}

template <class T>
T ** Sfann::add_matrix_x(T ** m, int x, T ** m2, int x2, int y) {
    if (m == NULL || x <= 0) {
        if (m2 != NULL && x2 > 0) {
            return copy_matrix<T>(m2, x2, y);
        } else {
            return NULL;
        }
    }
    if (m2 == NULL || x2 <= 0) {
        if (m != NULL && x > 0) {
            return copy_matrix<T>(m, x, y);
        } else {
            return NULL;
        }
    }
    
    T ** m3 = new T*[x+x2];

    for (int i=0; i<x; i++) {
        m3[i] = new T[y];
        for (int j=0; j<y; j++) {
            m3[i][j] = m[i][j];
        }
    }

    for (int i=x; i<x+x2; i++) {
        m3[i] = new T[y];
        for (int j=0; j<y; j++) {
            m3[i][j] = m2[i][j];
        }
    }

    return m2;
}

// struct fann_train_data
// {
// 	enum fann_errno_enum errno_f;
// 	FILE *error_log;
// 	char *errstr;
//
// 	unsigned int num_data;
// 	unsigned int num_input;
// 	unsigned int num_output;
// 	fann_type **input;
// 	fann_type **output;
// };

void Sfann::delete_fann_train_data(struct fann_train_data * & d, int nb) {
    if (d == NULL) return;
    for (int n = 0; n < nb; n++) {
        for (int i=0; i<d[n].num_data; i++) {
            delete[] d[n].input[i];
            delete[] d[n].output[i];
        }
        //if (d[n].errstr != NULL) delete[] d[n].errstr;
        //if (d[n].error_log != NULL) delete d[n].error_log;
    }
    delete[] d;
    d = NULL;
}

void Sfann::delete_train_dev_test_couple(train_dev_test_couple * & cross_corpora, int nb_copora) {
    if (cross_corpora == NULL) {return;}
    for (int k=0; k<nb_copora; k++) {
        //struct fann_train_data * p = &(cross_corpora[k].train);
        delete_fann_train_data(cross_corpora[k].train, 1);
        delete_fann_train_data(cross_corpora[k].dev, 1);
        delete_fann_train_data(cross_corpora[k].test, 1);
//     fann_destroy_train(&(cross_corpora[k].train));
//        fann_destroy_train(&(cross_corpora[k].dev));
//       fann_destroy_train(&(cross_corpora[k].test));
    }
    delete[] cross_corpora;
    cross_corpora = NULL;
}

void Sfann::delete_folds(folds * & f) {
    if (f == NULL) return;

    for (int k=0; k<f->num_folds; k++) {
        fann_destroy_train(&f->data[k]);
        /*for (int j=0; j<f->data[k].num_data; j++) {
            if (f->data[k].input[j] != NULL) delete[] f->data[k].input[j];
            if (f->data[k].output[j] != NULL) delete[] f->data[k].output[j];
            if (f->data[k].errstr != NULL) free(f->data[k].errstr);
        }*/
    }
    delete[] f->data;
    delete f;
    f = NULL;
}

train_dev_test_couple * create_train_dev_test_couple() {
    train_dev_test_couple * t = new train_dev_test_couple;
    t->train = NULL;
    t->dev = NULL;
    t->test = NULL;
    return t;
}

void Sfann::generate_folds_from_train_corpus(struct fann_train_data * train_data, folds & _folds, int cross_nb_folds) {
    int num_by_fold = train_data->num_data/cross_nb_folds + 1;
    cout << train_data->num_data << " into " << cross_nb_folds << " folds = " << num_by_fold << " / fold... ";

    _folds.data = new struct fann_train_data[cross_nb_folds];
    _folds.num_folds = cross_nb_folds;
    for (int i=0; i<cross_nb_folds; ++i) {
        init_structure_metadata(train_data, &(_folds.data[i]), num_by_fold);
    }

    int nb_copies = 0;
    while (nb_copies < train_data->num_data) {
        for (int i=0; i<cross_nb_folds; ++i) {
            copy_train_data(train_data, &(_folds.data[i]), nb_copies, 1);
            nb_copies++;
            if (nb_copies >= train_data->num_data) break;
        }
    }
    _folds.num_data = nb_copies;
}

// Cree un fann_train_data pour un nombre donne d'exemples (num_data), et utilise src pour copier les meta-donnees
void Sfann::init_structure_metadata(struct fann_train_data * src, struct fann_train_data * dest, int num_data) {
// 	cerr << "Init " << num_data << endl;

    if (src == NULL || dest == NULL) {
        return;
    }

    dest->errno_f = src->errno_f;
    dest->error_log = NULL; // src->error_log;
    dest->errstr = NULL; // src->errstr;
    dest->num_input = src->num_input;
    dest->num_output = src->num_output;
    dest->num_data = 0;

    if (num_data <= 0) {
        dest->input = NULL;
        dest->output = NULL;
    } else {
    // 	dest->input = new fann_type[num_data][num_in];
        dest->input = new fann_type * [num_data];
        for (int i=0; i<num_data; i++) dest->input[i] = new fann_type[src->num_input]; // (fann_type*) malloc(sizeof(fann_type)*src->num_input); // new fann_type[src->num_input];

    // 	dest->output = new fann_type[num_data][num_out];
        dest->output = new fann_type * [num_data];
        for (int i=0; i<num_data; i++) dest->output[i] = new fann_type[src->num_output];
    }
}

void Sfann::copy_train_data(struct fann_train_data * src, struct fann_train_data * dest) {
    copy_train_data(src, dest, 0, src->num_data);
}

void Sfann::copy_train_data(struct fann_train_data * src, struct fann_train_data * dest, int start, int nb) {
    if (src == NULL || dest == NULL) return;

    for (int i=start; i<start+nb; i++) {
        for (int k=0; k<src->num_input; k++) {
            dest->input[dest->num_data][k] = src->input[i][k];
        }
        for (int k=0; k<src->num_output; k++) {
            dest->output[dest->num_data][k] = src->output[i][k];
        }
        dest->num_data++;
    }
}



void Sfann::generate_cross_corpus(folds & _folds, train_dev_test_couple * cross_corpus, int num_test_fold, int nb_dev_folds) {

    if (cross_corpus == NULL) return;

    set<int> dev_folds;
    int num_dev_data = 0;
    int pos = num_test_fold + 1;
    for (int i=0; i<nb_dev_folds; i++) {
        if (pos >= _folds.num_folds) pos = 0;
        num_dev_data += _folds.data[pos].num_data;
        dev_folds.insert(pos);
        pos++;
    }

    printf(" test:%d dev:%d train:%d... ", _folds.data[num_test_fold].num_data, num_dev_data, _folds.num_data - _folds.data[num_test_fold].num_data - num_dev_data);

    cross_corpus->test = new struct fann_train_data;
    cross_corpus->dev = new struct fann_train_data;
    cross_corpus->train = new struct fann_train_data;

    init_structure_metadata(&(_folds.data[0]), cross_corpus->test, _folds.data[num_test_fold].num_data);
    init_structure_metadata(&(_folds.data[0]), cross_corpus->dev, num_dev_data);
    init_structure_metadata(&(_folds.data[0]), cross_corpus->train, _folds.num_data - _folds.data[num_test_fold].num_data - num_dev_data);

    for (int n=0; n<_folds.num_folds; n++) {
        if (n == num_test_fold) {
            // creation du corpus de test
            copy_train_data(&(_folds.data[n]), cross_corpus->test);
        } else if (dev_folds.count(n) > 0) {
            // creation du corpus de dev
            copy_train_data(&(_folds.data[n]), cross_corpus->dev);
        } else {
            // creation du corpus de train
            copy_train_data(&(_folds.data[n]), cross_corpus->train);
        }
    }
}

void Sfann::do_training() {

    training_res * res_global = NULL;

    if ((*this->config).count("do-nothing")) {
        return;
    } else if ((*this->config).count("do-cross-validation")) {
        int cross_nb_folds = 0;
        int cross_nb_dev = 0;

        bool cross_shuffle_data = (*this->config).count("cv-shuffle");
        if ((*this->config).count("cv-num-folds")) cross_nb_folds = (*this->config)["cv-num-folds"].as<int>();
        if ((*this->config).count("cv-num-dev")) cross_nb_dev = (*this->config)["cv-num-dev"].as<int>();
        if ((*this->config).count("leave-one-out")) cross_nb_folds = this->train_data->num_data;

        if (cross_nb_folds > 0) {
            cout << " ->  Creating cross-validation folds... ";
            folds cross_folds;
            if (cross_shuffle_data) fann_shuffle_train_data(this->train_data);
            this->generate_folds_from_train_corpus(this->train_data, cross_folds, cross_nb_folds);
            cout << "Ok !" << endl;

            //      cout << cross_folds.num_folds << endl;

            // sauvegarde du contexte
            struct fann_train_data * train_tmp = this->train_data, * dev_tmp = this->dev_data, * test_tmp = this->test_data;
            this->train_data = NULL; this->dev_data = NULL; this->test_data = NULL;
            res_global = this->create_training_res();

            for (int i=0; i<cross_nb_folds; ++i) {
                cout << " ->  Validation number " << i+1 << " ..." << endl;

                cout << "     - creating corpus couple ... "; fflush(stdout);
                train_dev_test_couple * cc = create_train_dev_test_couple();
                this->generate_cross_corpus(cross_folds, cc, i, cross_nb_dev);
                cout << "Ok !" << endl;

                cout << "     - training ... "; fflush(stdout);

                this->train_data = cc->train;
                if (cc->dev->num_data > 0) {
                    this->dev_data = cc->dev;
                } else {
                    this->dev_data = NULL;
                }
                if (cc->test->num_data > 0) {
                    this->test_data = cc->test;
                } else {
                    this->test_data = NULL;
                }

                training_res * res = this->do_normal_training(1);

                cout << "Ok !" << endl;

                printf("     - classif. rate for this iteration :\n");
                print_training_res(res);

                this->add_training_res(res, res_global);

                this->delete_train_dev_test_couple(cc, 1);
                this->train_data = NULL; this->dev_data = NULL; this->test_data = NULL;
                this->delete_training_res(res, 1);
            }

            // restauration du contexte
            this->train_data = train_tmp;
            this->dev_data = dev_tmp;
            this->test_data = test_tmp;

            printf(" => Overall classif. rate :\n");
            print_training_res(res_global);

//             this->delete_training_res(res_global, 1);
        }

    } else if ((*this->config).count("do-training")) {
        res_global = this->do_normal_training(1);

        // save the desired ann
        if ((*this->config).count("save-max-dev") && res_global->net_max_dev->net != NULL) {
            fann_save(res_global->net_max_dev->net, (*this->config)["save-max-dev"].as<string>().c_str());
        }
        if ((*this->config).count("save-max-test") && res_global->net_max_test->net != NULL) {
            fann_save(res_global->net_max_test->net, (*this->config)["save-max-test"].as<string>().c_str());
        }
        if ((*this->config).count("save-max-train") && res_global->net_max_train->net != NULL) {
            fann_save(res_global->net_max_train->net, (*this->config)["save-max-train"].as<string>().c_str());
        }

        if (this->test_data != NULL && ((*this->config).count("save-max-dev-run") || (*this->config).count("save-max-test-run") || (*this->config).count("save-max-train-run"))) {
            // save the desired results
            fann_train_data * tmp = new fann_train_data;
            init_structure_metadata(this->test_data, tmp, 0);
            tmp->input = this->test_data->input;
            tmp->num_data = this->test_data->num_data;

            if ((*this->config).count("save-max-dev-run") && res_global->net_max_dev->test_out != NULL) {
                tmp->output = res_global->net_max_dev->test_out;
                fann_save_train(tmp, (*this->config)["save-max-dev-run"].as<string>().c_str());
            }

            if ((*this->config).count("save-max-test-run") && res_global->net_max_test->test_out != NULL) {
                tmp->output = res_global->net_max_test->test_out;
                fann_save_train(tmp, (*this->config)["save-max-test-run"].as<string>().c_str());
            }

            if ((*this->config).count("save-max-train-run") && res_global->net_max_train->test_out != NULL) {
                tmp->output = res_global->net_max_train->test_out;
                fann_save_train(tmp, (*this->config)["save-max-train-run"].as<string>().c_str());
            }
            
            delete tmp;
        }
        
    } else if ((*this->config).count("do-running")) {
        printf("-> Loading %s ...", (*this->config)["load-ann"].as<string>().c_str());
        struct fann * net = fann_create_from_file((*this->config)["load-ann"].as<string>().c_str());
        printf(" Ok !\n");
        
        int nb_ok = 0;
        fann_type ** output = NULL;
        perfs_on_data(net, this->test_data, nb_ok, output);

        printf("-> Classification rate on the test data : %.2f\n", (float)nb_ok*100/(float)this->test_data->num_data);

        if ((*this->config).count("save-loaded-run")) {
            fann_train_data * tmp = new fann_train_data;
            init_structure_metadata(this->test_data, tmp, 0);
            tmp->input = this->test_data->input;
            tmp->output = output;
            tmp->num_data = this->test_data->num_data;

            fann_save_train(tmp, (*this->config)["save-loaded-run"].as<string>().c_str());

            delete tmp;
        }

    }

    this->delete_training_res(res_global, 1);
}


training_res * Sfann::do_normal_training(int detail) {
    int num_input = fann_num_input_train_data(this->train_data);
    int num_output = fann_num_output_train_data(this->train_data);
    int num_hidden = (*this->config)["num-hidden"].as<int>();
    int num_runs = (*this->config)["num-runs"].as<int>();
    int max_epochs = (*this->config)["max-epoch"].as<int>();
    int num_reports = (*this->config)["reports"].as<int>();
    float desired_error = (*this->config)["desired-error"].as<float>();
    bool randomize = (*this->config).count("randomize-data");
    bool clever_init = (*this->config).count("clever-init");

    if (detail > 0) cout << " ->  Network has 3 layers : " << num_input << "->" << num_hidden << "->" << num_output << endl;
    if (detail > 0) cout << " ->  Training on " << this->train_data->num_data << " data (" << this->train_data->num_input << "->" << this->train_data->num_output << ")" << endl;
    if (randomize && detail > 0) cout << " ->  Training data are shuffled on each run" << endl;
    if (clever_init && detail > 0) cout << " ->  Network weights are initialized using the Widrow + Nguyen's algorithm" << endl;

    this->train_courant = this->create_training_res();

/*    float somme_perfs = 0.;
    int somme_num_ok = 0;*/
    for (int k=0; k<num_runs; k++) {

        if (detail > 0) cout << " ->  Iteration " << k << endl;


/*        if (this->train_courant != NULL) {
            this->delete_training_res(this->train_courant, 1);
        }

        this->train_courant = this->create_training_res();
*/


// 		struct fann * net = fann_create_standard(3, num_input, num_output, num_hidden);
        struct fann * net = fann_create_sparse(1.0, 3, num_input, num_hidden, num_output);

        fann_set_training_algorithm(net, FANN_TRAIN_RPROP);

        fann_set_activation_function_hidden(net, FANN_SIGMOID_SYMMETRIC);
        fann_set_activation_function_output(net, FANN_SIGMOID_SYMMETRIC);

        fann_set_train_error_function(net, FANN_ERRORFUNC_LINEAR);

        fann_set_train_stop_function(net, FANN_STOPFUNC_MSE);

        fann_set_learning_rate(net, 0.7);
        fann_set_learning_momentum(net, 0.0);

        fann_set_activation_steepness_hidden(net,(fann_type) 0.5);
        fann_set_activation_steepness_output(net,(fann_type) 0.5);

        fann_set_quickprop_decay(net, -0.0001);
        fann_set_quickprop_mu(net, 1.75);

        fann_set_rprop_increase_factor(net,1.2);
        fann_set_rprop_decrease_factor(net,0.5);
        fann_set_rprop_delta_min(net,0);
        fann_set_rprop_delta_max(net,50);


        if (randomize) {
            fann_shuffle_train_data(this->train_data);
        }

        if (clever_init) {
            fann_init_weights(net, this->train_data);
        }


        fann_set_callback(net, training_callback);

        fann_train_on_data(net, this->train_data, max_epochs, num_reports, desired_error);

        if (detail > 0) print_training_res(this->train_courant);

        

// 		cout<<"Iteration " << k << " : score=" << score*100 << " best_dev=" << this->best_dev->dev_perfs << " " << endl;
    }

/*    res->avg_num_ok = somme_num_ok/num_runs;
    res->avg_perfs = somme_perfs/num_runs;

    res->num_data = this->dev_data->num_data;*/

    //if (detail > 0) printf(" => Globaf classif : max = %.2f %% ; avg = %.2f %%\n", res->max_perfs*100, res->avg_perfs*100);

    if (detail > 0) printf("-> Training done !\n");
    
    training_res * res = this->train_courant;
    this->train_courant = NULL;

    return res;
}





